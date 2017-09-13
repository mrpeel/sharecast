'use strict';

const yahooFinance = require('yahoo-finance');
const utils = require('./utils');
const dynamodb = require('./dynamodb');
// const retrieveData = require('./dynamo-retrieve-share-data');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const moment = require('moment-timezone');
const sns = require('./publish-sns');
const snsArn = 'arn:aws:sns:ap-southeast-2:815588223950:lambda-activity';
const aws = require('aws-sdk');
const lzwCompress = require('lzwcompress');

const lambda = new aws.Lambda({
  region: 'ap-southeast-2',
});

/**  Reload all historical adjusted prices for a symbol
*    Retrieves the complete history of a symbol and updates the adjustedPrice -
*     triggered by a stock / split or other adjustment which causes historical
*     adjusted prices to be updated
* @param {Object} params - {
  endDate - date to end lookup
  symbol - symbol to retrieve
}
*/
let retrieveAdjustedHistoryData = asyncify(function(params) {
  try {
    if (!params.symbol) {
      console.error(`Symbol parameter not provided`);
      return;
    }

    if (!params.endDate || !utils.isDate(params.endDate)) {
      console.error(`Invalid params - valid endDate required: `,
        `${params.endDate}`);
      return;
    }

    let t0 = new Date();

    let symbol = params.symbol;
    let companySymbol = awaitify(getCompanySymbol(symbol));
    let results;
    let maxLength = 100;
    let endDate = params.endDate;
    let fullResults = awaitify(retrieveHistory(companySymbol,
      '2007-07-01', utils.returnDateAsString(endDate, 'YYYY-MM-DD')));

    results = [];

    // Strip unnecessary information from results
    fullResults.forEach((resultVal) => {
      // Push the stripped down value to results
      results.push({
        'date': utils.returnDateAsString(resultVal['date']),
        'adjClose': resultVal['adjClose'],
      });

      // Check if number of results has been reached to execute lambda
      if (results.length >= maxLength) {
        let invokeEvent = {
          'symbol': symbol,
          'results': results,
        };

        let description = `Invoking reloadAdjustedPrices:  ${symbol} - ` +
          `${results.length} records`;

        awaitify(invokeLambda('reloadAdjustedPrices', invokeEvent, description));

        // Reset results to empty array
        results = [];
      }
    });
    return true;
  } catch (err) {
    try {
      awaitify(
        sns.publishMsg(snsArn,
          'retrieveAdjustedHistoryData failed.  Error: ' + JSON.stringify(err),
          'retrieveAdjustedHistoryData failed'));
    } catch (err) {}

    console.log('retrieveAdjustedHistoryData failed while processing: ', params.symbol);
    console.log(err);
  }
});

/**  Execute updates for an array of symbol, date, adjclose combinations
*/
let reloadAdjustedPrices = asyncify(function(params) {
  try {
    if (!params.symbol) {
      console.error(`Symbol parameter not provided`);
      return;
    }

    if (params.results) {
      console.error(`Invalid params - results not provided `);
      return;
    }

    let t0 = new Date();

    let symbol = params.symbol;
    let results = params.results;

    while (results.length) {
      let result = results.shift(1);
      result.adjustedPrice = result.adjClose;
      result.symbol = symbol;
      awaitify(updateAdjustedPrice(result));

      let t1 = new Date();

      if (utils.dateDiff(t0, t1, 'seconds') > 250) {
        break;
      }
    }

    /*  If results is not empty, more procesing is required
         re-invoke lambda with remaining results */
    if (results.length) {
      let reinvokeEvent = {
        'symbol': symbol,
        'results': results,
      };

      let description = `Continuing reloadAdjustedPrices for ${symbol}. ` +
        `${results.length} still to complete`;

      awaitify(invokeLambda('reloadAdjustedPrices', reinvokeEvent, description));
    } else {
      awaitify(
        sns.publishMsg(snsArn, 'reloadAdjustedPrices completed.',
          `reloadAdjustedPrices for ${symbol} completed.`));
    }

    return true;
  } catch (err) {
    try {
      awaitify(
        sns.publishMsg(snsArn,
          'reloadAdjustedPrices failed.  Error: ' + JSON.stringify(err),
          `reloadAdjustedPrices failed for ${params.symbol}`));
    } catch (err) {}

    console.error('reloadAdjustedPrices failed while processing: ', params.symbol);
    console.error(err);
  }
});

let retrieveHistory = function(symbol, startDate, endDate) {
  return new Promise(function(resolve, reject) {
    let historyOptions = {
      from: startDate,
      to: endDate,
      symbol: symbol,
    };

    yahooFinance.historical(historyOptions).then(function(result) {
      resolve(result);
    }).catch(function(err) {
      reject(err);
    });
  });
};

let getCompanySymbol = function(symbol) {
  return new Promise(function(resolve, reject) {
    try {
      let queryDetails = {
        tableName: 'companies',
        keyConditionExpression: 'symbol = :symbol',
        expressionAttributeValues: {
          ':symbol': symbol,
        },
        reverseOrder: true,
        limit: 1,
      };

      let symbolResults = awaitify(dynamodb.queryTable(queryDetails));

      if (symbolResults.length) {
        resolve(symbolResults[0]['symbolYahoo']);
      } else {
        reject(`Symbol ${symbol} not found`);
      }
    } catch (err) {
      console.log(err);
      reject(err);
    }
  });
};

/**  Write a conpany quote adjusted price update to dynamodb
* @param {Object} quoteData - the company quote to write
*/
let updateAdjustedPrice = asyncify(function(quoteData) {
  // If unexpected r3cords come back with no trade data, skop them
  if (!quoteData['date']) {
    return;
  }

  let updateDetails = {
    tableName: 'companyQuotes',
  };

  quoteData.quoteDate = utils.returnDateAsString(quoteData['date']);
  if (!quoteData.adjustedPrice) {
    return;
  }

  updateDetails.key = {
    symbol: quoteData.symbol,
    quoteDate: quoteData.quoteDate,
  };

  let updateExpression;
  let expressionAttributeValues = {};
  let expressionAttributeNames = {};

  // Separately set-up creeated attributed values and names
  expressionAttributeValues[':updated'] = moment().tz('Australia/Sydney').format();
  expressionAttributeNames['#updated'] = 'updated';

  expressionAttributeValues[':adjustedPrice'] = quoteData['adjustedPrice'];
  expressionAttributeNames['#adjustedPrice'] = 'adjustedPrice';

  updateExpression = 'set #adjustedPrice = :adjustedPrice';

  updateDetails.updateExpression = updateExpression;
  updateDetails.expressionAttributeValues = expressionAttributeValues;
  updateDetails.expressionAttributeNames = expressionAttributeNames;

  return awaitify(dynamodb.updateRecord(updateDetails));
});

let invokeLambda = function(lambdaName, event, description) {
  return new Promise(function(resolve, reject) {
    console.log(`Invoking lambda ${lambdaName} with event: ${event}`);
    if (description) {
      console.log(description);
    }

    /*
    // Re-invoke function to simulate lambda
    retrieveAdjustedHistoryData(event);

    resolve(true);
    return; */

    lambda.invoke({
      FunctionName: lambdaName,
      InvocationType: 'Event',
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
  retrieveAdjustedHistoryData: retrieveAdjustedHistoryData,
  reloadAdjustedPrices: reloadAdjustedPrices,
};

// let testLoad = asyncify(function() {
//   dynamodb.setLocalAccessConfig();
//
//   awaitify(retrieveAdjustedHistoryData({
//     endDate: '2017-09-10',
//     symbol: 'WTC',
//   }));
// });
//
// testLoad();
