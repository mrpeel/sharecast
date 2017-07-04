const yahooFinance = require('yahoo-finance');
const utils = require('../retrieve-data/utils');
const dynamodb = require('../retrieve-data/dynamodb');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const moment = require('moment-timezone');
const sns = require('./publish-sns');
const snsArn = 'arn:aws:sns:ap-southeast-2:815588223950:lambda-activity';
const aws = require('aws-sdk');

const lambda = new aws.Lambda({
  region: 'ap-southeast-2',
});

/**  Reload all historical adjusted prices for a symbol
*    Retrieves the complete history of a symbol and updates the adjustedPrice -
*     triggered by a stock / split or other adjustment which causes historical
*     adjusted prices to be updated
* @param {String} symbol - the company to reload
* @param {String} endDate - the the date to retrieve up to
*/
let reloadAdjustedPrices = asyncify(function(params) {
  try {
    if (!params.symbol) {
      console.error(`Symbol parameter not provided`);
      return;
    }

    if ((!params.endDate || !utils.isDate(params.endDate)) && !params.results) {
      console.error(`Invalid params - valid endDate or results required: `,
        `${params.endDate}, ${params.results}`);
      return;
    }

    let t0 = new Date();

    let symbol = params.symbol;
    let companySymbol = awaitify(getCompanySymbol(symbol));
    let results;

    if (params.results) {
      results = params.results;
    } else {
      let endDate = params.endDate;
      results = awaitify(retrieveHistory(companySymbol,
        '2008-07-01', utils.returnDateAsString(endDate, 'YYYY-MM-DD')));
    }

    for (let c = 0; c < results.length; c++) {
      let result = results.shift(1);
      result.adjustedPrice = result.adjClose;
      result.symbol = symbol;
      awaitify(updateAdjustedPrice(result));

      let t1 = new Date();

      if (utils.dateDiff(t0, t1, 'seconds') > 250) {
        let reinvokeEvent = {
          'symbol': symbol,
          'results': results,
        };

        invokeLambda('reloadAdjustedPrices', reinvokeEvent);
        break;
      }
    }

    return true;
  } catch (err) {
    try {
      awaitify(
        sns.publishMsg(snsArn,
          'reloadAdjustedPrices failed.  Error: ' + JSON.stringify(err),
          'reloadAdjustedPrices failed'));
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

  updateExpression = 'set #adjustedPrice = :adjustedPrice' +
    ', #updated = :updated';

  updateDetails.updateExpression = updateExpression;
  updateDetails.expressionAttributeValues = expressionAttributeValues;
  updateDetails.expressionAttributeNames = expressionAttributeNames;

  return awaitify(dynamodb.updateRecord(updateDetails));
});

let invokeLambda = function(lambdaName, event) {
  return new Promise(function(resolve, reject) {
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
  reloadAdjustedPrices: reloadAdjustedPrices,
};
