const utils = require('./utils');
const dynamodb = require('./dynamodb');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const retrieveData = require('./dynamo-retrieve-share-data');
const processRollups = require('./process-rollups');
const sns = require('./publish-sns');
const snsArn = 'arn:aws:sns:ap-southeast-2:815588223950:lambda-activity';
const aws = require('aws-sdk');

const lambda = new aws.Lambda({
  region: 'ap-southeast-2',
});

let processReturns = asyncify(function(event) {
  let endDate = event.endDate || utils.returnDateAsString(Date.now());
  let symbols;

  let t0 = new Date();

  // If continuing used supplied symbolGroups, otherwise create the groups
  if (event.symbols) {
    symbols = event.symbols;
  } else {
    let symbolResult = awaitify(retrieveData.setupSymbols());
    symbols = [];

    Object.keys(symbolResult.companyLookup).forEach((companySymbol) => {
      symbols.push(symbolResult.companyLookup[companySymbol]);
    });
  }


  while (symbols.length) {
    try {
      let symbol = symbols.shift(1);
      let processingData = awaitify(getCurrentAndPrevious52WeekPrices(symbol,
        endDate));
      awaitify(processCompanyReturns(symbol, processingData.currentWeekPrices,
        processingData.historyPrices, processingData.dividends,
        processingData.lastBollingerValues));
    } catch (err) {
      console.error(err);
    }
    let t1 = new Date();

    if (utils.dateDiff(t0, t1, 'seconds') > 250) {
      break;
    }
  }

  /*  If symbols is not empty, more procesing is required
       re-invoke lambda with remaining symbols */
  if (symbols.length) {
    let reinvokeEvent = {
      'symbols': symbols,
      'endDate': endDate,
    };

    let description = `Continuing processReturns. ` +
      `${symbols.length} symbols still to complete`;

    awaitify(invokeLambda('processReturns', reinvokeEvent, description));
  } else {
    awaitify(
      sns.publishMsg(snsArn, 'processReturns completed.',
        'processReturns completed.'));
  }

  return true;
});

let getCurrentAndPrevious52WeekPrices = asyncify(function(symbol,
  referenceDate) {
  let historyPrices = {};
  let currentWeekPrices = {};
  let dividends = {};
  let lastBollingerValues = {};
  let endDate = utils.dateAdd(referenceDate, 'days', -1);
  let currentWeekStart = utils.dateAdd(referenceDate, 'weeks', -1);
  let startDate = utils.dateAdd(referenceDate, 'weeks', -53);

  let queryDetails = {
    tableName: 'companyQuotes',
    keyConditionExpression: 'symbol = :symbol and ' +
      'quoteDate between :startDate and :endDate',
    filterExpression: 'attribute_exists(adjustedPrice)',
    expressionAttributeValues: {
      ':startDate': startDate,
      ':endDate': endDate,
    },
    projectionExpression: 'symbol, quoteDate, #adjustedPrice, ' +
      '#exDividendDate, #exDividendPayout, #4WeekBollingerType, ' +
      '#12WeekBollingerType',
    expressionAttributeNames: {
      '#adjustedPrice': 'adjustedPrice',
      '#exDividendDate': 'exDividendDate',
      '#exDividendPayout': 'exDividendPayout',
      '#4WeekBollingerType': '4WeekBollingerType',
      '#12WeekBollingerType': '12WeekBollingerType',
    },
  };

  queryDetails.expressionAttributeValues[':symbol'] = symbol;

  let queryResults = awaitify(dynamodb.queryTable(queryDetails));

  queryResults.forEach((result) => {
    let adjustedPrice = result.adjustedPrice;
    let quoteDate = result.quoteDate;

    // Add current week prices if in the current period
    if (quoteDate >= currentWeekStart) {
      currentWeekPrices[quoteDate] = adjustedPrice;
    }

    // All values go into history
    historyPrices[quoteDate] = adjustedPrice;

    // Check if there is a new dividend within the date period
    if (result.exDividendDate && result.exDividendPayout &&
      result.exDividendDate >= startDate &&
      result.exDividendDate < currentWeekStart) {
      // Check whether this dividend already exists
      if (!dividends[result.exDividendDate]) {
        dividends[result.exDividendDate] = result.exDividendPayout;
      }
    }

    // Check if there is an updated Bollinger type
    // Add 4 Week type if it is missing and the current quote date is before
    // the start of this period and it is newer than anything already recorded
    if (result['4WeekBollingerType'] && quoteDate < currentWeekStart &&
      (!lastBollingerValues.last4Week ||
      (quoteDate > lastBollingerValues.last4Week.date))) {
      lastBollingerValues.last4Week = {
        date: quoteDate,
        type: result['4WeekBollingerType'],
      };
    }
    // Add 12 Week type if it is missing and the current quote date is before
    // the start of this period and it is newer than anything already recorded
    if (result['12WeekBollingerType'] && quoteDate < currentWeekStart &&
      (!lastBollingerValues.last12Week ||
      (quoteDate > lastBollingerValues.last12Week.date))) {
      lastBollingerValues.last12Week = {
        date: quoteDate,
        type: result['12WeekBollingerType'],
      };
    }
  });

  return {
    currentWeekPrices: currentWeekPrices,
    historyPrices: historyPrices,
    dividends: dividends,
    lastBollingerValues: lastBollingerValues,
  };
});


let processCompanyReturns = asyncify(function(symbol, currentWeekPrices,
  historicalPrices, dividends, bollingerLastValues) {
  try {
    Object.keys(currentWeekPrices).forEach((returnDate) => {
      // for (let c = 0; c < Object.keys(currentWeekPrices).length; c++) {
      // let returnDate = Object.keys(currentWeekPrices)[c];
      let updateReturns = {};
      let price = currentWeekPrices[returnDate];

      updateReturns.symbol = symbol;
      updateReturns.quoteDate = returnDate;

      // Calculate std deviation for 1, 2, 4, 8, 12, 26, 52 weeks
      let weekStats = processRollups.getWeeklyStats(
        historicalPrices,
        returnDate);

      updateReturns['1WeekVolatility'] = weekStats['1WeekStdDev'];
      updateReturns['2WeekVolatility'] = weekStats['2WeekStdDev'];
      updateReturns['4WeekVolatility'] = weekStats['4WeekStdDev'];
      updateReturns['8WeekVolatility'] = weekStats['8WeekStdDev'];
      updateReturns['12WeekVolatility'] = weekStats['12WeekStdDev'];
      updateReturns['26WeekVolatility'] = weekStats['26WeekStdDev'];
      updateReturns['52WeekVolatility'] = weekStats['52WeekStdDev'];


      updateReturns['4WeekBollingerBandUpper'] = weekStats['4WeekBollingerBandUpper'];
      updateReturns['4WeekBollingerBandLower'] = weekStats['4WeekBollingerBandLower'];
      updateReturns['4WeekBollingerPrediction'] = 'Steady';
      if (price > weekStats['4WeekBollingerBandUpper']) {
        updateReturns['4WeekBollingerType'] = 'Above';
      } else if (price < weekStats['4WeekBollingerBandLower']) {
        updateReturns['4WeekBollingerType'] = 'Below';
      } else {
        updateReturns['4WeekBollingerType'] = 'Within';
      }


      updateReturns['12WeekBollingerBandUpper'] = weekStats['12WeekBollingerBandUpper'];
      updateReturns['12WeekBollingerBandLower'] = weekStats['12WeekBollingerBandLower'];
      updateReturns['12WeekBollingerPrediction'] = 'Steady';
      if (price > weekStats['12WeekBollingerBandUpper']) {
        updateReturns['12WeekBollingerType'] = 'Above';
      } else if (price < weekStats['12WeekBollingerBandLower']) {
        updateReturns['12WeekBollingerType'] = 'Below';
      } else {
        updateReturns['12WeekBollingerType'] = 'Within';
      }


      // Check for movements down from above upper band and up from below lower band
      if (bollingerLastValues.last4Week) {
        let last4Week = bollingerLastValues.last4Week;

        if (last4Week === 'Above' &&
          updateReturns['4WeekBollingerType'] === 'Within') {
          updateReturns['4WeekBollingerPrediction'] = 'Falling';
        } else if (last4Week === 'Below' &&
          updateReturns['4WeekBollingerType'] === 'Within') {
          updateReturns['4WeekBollingerPrediction'] = 'Rising';
        }
      }

      if (bollingerLastValues.last12Week) {
        let last12Week = bollingerLastValues.last12Week;
        if (last12Week === 'Above'
          && updateReturns['12WeekBollingerType'] === 'Within') {
          updateReturns['12WeekBollingerPrediction'] = 'Falling';
        } else if (last12Week === 'Below' &&
          updateReturns['12WeekBollingerType'] === 'Within') {
          updateReturns['12WeekBollingerPrediction'] = 'Rising';
        }
      }

      bollingerLastValues.last4Week = {
        'date': returnDate,
        'type': updateReturns['12WeekBollingerType'],
      };

      bollingerLastValues.last12Week = {
        'date': returnDate,
        'type': updateReturns['12WeekBollingerType'],
      };

      // Update stats information into today's record
      awaitify(updateCompanyQuoteData(updateReturns));

      /* Calculate and update total return and risk adjusted return
        for 1, 2, 4, 8, 12, 26, 52 weeks */
      awaitify(processRollups.updateReturns(
        symbol,
        returnDate,
        price,
        historicalPrices,
        dividends || {}));
    });
  // }
  } catch (err) {
    try {
      awaitify(
        sns.publishMsg(snsArn,
          'processCompanyReturns failed.  Error: ' + JSON.stringify(err),
          'processCompanyReturns failed whilte processing ' + symbol));
    } catch (err) {}

    console.error('processCompanyReturns failed while processing: ', symbol);
    console.error(err);
  }
});

/**  Write a conpany quote record to dynamodb.
*    Converts lastTradeDate -> quoteDate, copies lastTradePriceOnly ->
*     adjustedPrice, and checks and removes any invalid values.
* @param {Object} quoteData - the company quote to write
*/
let updateCompanyQuoteData = asyncify(function(quoteData) {
  // If unexpected r3cords come back with no trade data, skop them
  let updateDetails = {
    tableName: 'companyQuotes',
  };

  // Check through for values with null and remove from object
  Object.keys(quoteData).forEach((field) => {
    if (quoteData[field] === null || quoteData[field] === '') {
      delete quoteData[field];
    }
  });

  updateDetails.key = {
    symbol: quoteData.symbol,
    quoteDate: quoteData.quoteDate,
  };

  let updateExpression;
  let expressionAttributeValues = {};
  let expressionAttributeNames = {};
  let fieldsPresent = [];

  delete quoteData.symbol;
  delete quoteData.quoteDate;

  // Get a list of fields and values to copy
  Object.keys(quoteData).forEach((field) => {
    expressionAttributeValues[(':' + field)] = quoteData[field];
    expressionAttributeNames[('#' + field)] = field;
    fieldsPresent.push('#' + field + '=:' + field);
  });

  // Enure that at least one field is present to update
  if (fieldsPresent.length) {
    updateExpression = 'set ' + fieldsPresent.join(',');

    updateDetails.updateExpression = updateExpression;
    updateDetails.expressionAttributeValues = expressionAttributeValues;
    updateDetails.expressionAttributeNames = expressionAttributeNames;

    let result = awaitify(dynamodb.updateRecord(updateDetails));
    if (result === 'skipped') {
      console.error(`Quote update skipped for ${JSON.stringify(updateDetails.key)}`);
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
});

let invokeLambda = function(lambdaName, event, description) {
  return new Promise(function(resolve, reject) {
    console.log(`Invoking lambda ${lambdaName} with event: ${JSON.stringify(event)}`);
    if (description) {
      console.log(description);
    }
    /* if (lambdaName === 'processReturns') {
      processReturns(event);
    }
    resolve(true);
    return; */

    lambda.invoke({
      FunctionName: lambdaName,
      Payload: JSON.stringify(event, null, 2),
    }, function(err, data) {
      if (err) {
        reject(err);
      } else {
        console.log(`Function ${lambdaName} executed with event: `,
          `${JSON.stringify(event)}`);
        resolve(true);
      }
    });
  });
};

module.exports = {
  processReturns: processReturns,
};

/* dynamodb.setLocalAccessConfig();
processReturns({
  endDate: '2017-02-11',
}); */
