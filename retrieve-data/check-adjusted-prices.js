'use strict';

const utils = require('./utils');
const retrieveData = require('./dynamo-retrieve-share-data');
const yahooFinance = require('yahoo-finance');
const dynamodb = require('./dynamodb');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const sns = require('./publish-sns');
const snsArn = 'arn:aws:sns:ap-southeast-2:815588223950:lambda-activity';
const aws = require('aws-sdk');
// const dynamodb = require('./dynamodb');


const lambda = new aws.Lambda({
  region: 'ap-southeast-2',
});


/** Retrieve daily history values for symbol(s) from yahoo finance
 * @param {Array} symbol - one or more yahoo symbols to lookup
 * @param {String} startDate - the first day of the lookup period
 * @param {String} endDate - the last day of the lookup period
 * @return {Promise} resolves with the history record in form:
 *    {
 *    date: 20 Jan 2017,
 *    open: 28.77,
 *    high: 28.84,
 *    low: 28.42,
 *    close: 28.81,
 *    volume: 536800,
 *    adjClose: 27.73,
 *    symbol: 'JBH.AX'
 * }
 */
let retrieveDailyHistory = function(symbol, startDate, endDate) {
  return new Promise(function(resolve, reject) {
    let historyOptions = {
      from: startDate,
      to: endDate,
      period: 'd',
    };

    // Check if one or many symbols
    if (Array.isArray(symbol)) {
      historyOptions.symbols = symbol;
    } else {
      historyOptions.symbol = symbol;
    }

    yahooFinance.historical(historyOptions).then(function(result) {
      resolve(result);
    }).catch(function(err) {
      reject(err);
    });
  });
};

/** Check through each company symbol, retrieve the last 8 days history data
 *    and see if any prices have closePrice !== adjustedPrice.  If a discrepancy
 *    is found, then all prices for that symbol need to be re-checked back to
 *    2006-07-01, so trigger a complete re-check of prices.
 *
 */
let checkForAdjustments = asyncify(function(event) {
  let endDate = event.compDate || utils.returnDateAsString(Date.now());
  /* This assumes process is being run on a Friday night and the look back
      period allows going back over the last days of the previous week */
  let startDate = utils.dateAdd(endDate, 'days', -8);
  let symbolResult = awaitify(retrieveData.setupSymbols());

  let companies = symbolResult.companies;
  let symbolLookup = symbolResult.symbolLookup;
  let symbolGroups = [];
  let startYear = 2007;

  let t0 = new Date();

  // If continuing used supplied symbolGroups, otherwise create the groups
  if (event.symbolGroups) {
    symbolGroups = event.symbolGroups;
  } else {
    symbolGroups = [];
    /* Split companies into groups of 15 to ensure request doesn't exceed api
        url length */
    for (let companyCounter = 0; companyCounter < companies.length; companyCounter += 20) {
      symbolGroups.push(companies.slice(companyCounter, companyCounter + 20));
    }
  }

  let insertDetails = {
    tableName: 'quoteReload',
    values: {},
    primaryKey: [
      'symbolYear',
    ],
  };


  while (symbolGroups.length) {
    let symbolGroup = symbolGroups.shift(1);

    try {
      let results = awaitify(retrieveDailyHistory(symbolGroup,
        startDate, endDate));

      Object.keys(results).forEach((resultSymbol) => {
        // Retrieve individual result
        let result = results[resultSymbol];
        /* If the adjusted price differs from the close price on any of the days
           then adjustments have happened and the entire history needs to be
           re-retrieved */
        if (result.some((historyRecord) => {
            if (historyRecord.adjClose && historyRecord.close) {
              return (historyRecord.adjClose !== historyRecord.close);
            } else {
              return false;
            }
          })) {
          // Retrieve original symbol from yahoo symbol
          let retrieveSymbol = symbolLookup[resultSymbol];
          let reloadYear = startYear;

          while (String(reloadYear) < endDate) {
            let symbolYear = retrieveSymbol + reloadYear;
            let reloadStartDate = String(reloadYear) + '-01-01';
            let reloadEndDate = String(reloadYear) + '-12-31';

            if (reloadEndDate > endDate) {
              reloadEndDate = endDate;
            }

            insertDetails.values = {
              'symbolYear': symbolYear,
              'symbol': retrieveSymbol,
              'yahooSymbol': resultSymbol,
              'startDate': reloadStartDate,
              'endDate': reloadEndDate,
            };
            awaitify(dynamodb.insertRecord(insertDetails));

            reloadYear += 1;
          }
        }
      });
    } catch (err) {
      console.error(err);
    }
    let t1 = new Date();

    if (utils.dateDiff(t0, t1, 'seconds') > 220) {
      break;
    }
  }

  /*  If results is not empty, more procesing is required
       re-invoke lambda with remaining symbolGroups */
  if (symbolGroups.length) {
    let reinvokeEvent = {
      'symbolGroups': symbolGroups,
      'compDate': endDate,
    };

    let description = `Continuing checkForAdjustments. ` +
      `${symbolGroups.length} symbol groups still to complete`;

    awaitify(invokeLambda('checkForAdjustments', reinvokeEvent, description));
  } else {
    let description = `Completed checkForAdjustments. Executing reloadQuote`;
    awaitify(invokeLambda('reloadQuote', {}, description));

    awaitify(
      sns.publishMsg(snsArn, 'checkForAdjustments completed.',
        'checkForAdjustments completed.'));
  }

  return true;
});

let invokeLambda = function(lambdaName, event, description) {
  return new Promise(function(resolve, reject) {
    console.log(`Invoking lambda ${lambdaName} with event: ${JSON.stringify(event)}`);
    if (description) {
      console.log(description);
    }

    /*
    // Simulate lambda invoke
    if (lambdaName === 'checkForAdjustments') {
      checkForAdjustments(event);
    } else if (lambdaName === 'retrieveAdjustedHistoryData') {
      retrieveAdjustedData.retrieveAdjustedHistoryData(event);
    }

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
  checkForAdjustments: checkForAdjustments,
};

// dynamodb.setLocalAccessConfig();
// checkForAdjustments({
//   compDate: '2017-09-08',
// });
