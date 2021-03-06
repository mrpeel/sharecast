'use strict';

const utils = require('./utils');
const dynamodb = require('./dynamodb');
const retrieveQuote = require('./retrieve-quote');
const retrieveData = require('./dynamo-retrieve-share-data');
const sns = require('./publish-sns');
const snsArn = 'arn:aws:sns:ap-southeast-2:815588223950:lambda-activity';
const aws = require('aws-sdk');


const lambda = new aws.Lambda({
  region: 'ap-southeast-2',
});


/** Check through each company symbol, retrieve the last 8 days history data
 *    and see if any prices have closePrice !== adjustedPrice.  If a discrepancy
 *    is found, then all prices for that symbol need to be re-checked back to
 *    2006-07-01, so trigger a complete re-check of prices.
 *  @param {Object} event - lambda event trigger
 */
let checkForAdjustments = async function(event) {
  let endDate = event.compDate || utils.returnDateAsString(Date.now());
  /* This assumes process is being run on a Friday night and the look back
      period allows going back over the last days of the previous week */
  let startDate = utils.dateAdd(endDate, 'days', -8);
  let symbolResult = await retrieveData.setupSymbols();

  let companies = symbolResult.companies;
  let symbolLookup = symbolResult.symbolLookup;
  let symbolGroups = [];
  let startYear = 2007;
  let processStats;

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

  if (event.processStats) {
    processStats = event.processStats;
  } else {
    processStats = {
      symbolsChecked: 0,
      symbolReloadRecordsCreated: 0,
    };
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
      let adjustmentResults = await retrieveQuote.checkForAdjustedPrices({
        symbols: symbolGroup,
        startDate: startDate,
        endDate: endDate,
      });

      console.log(`Number of succesfull results: ${JSON.stringify(Object.keys(adjustmentResults.results).length)}`);
      console.log(`Number of errors: ${adjustmentResults.errorCount}`);


      for (const adjustmentResult of adjustmentResults.results) {
        // Retrieve individual result
        if (adjustmentResult.adjustedResult) {
          // Retrieve original symbol from yahoo symbol
          let retrieveSymbol = symbolLookup[adjustmentResult.symbol];
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
              'yahooSymbol': adjustmentResult.symbol,
              'startDate': reloadStartDate,
              'endDate': reloadEndDate,
            };
            await dynamodb.insertRecord(insertDetails);
            processStats.symbolReloadRecordsCreated++;

            reloadYear += 1;
          }
        }
        processStats.symbolsChecked++;
      }
    } catch (err) {
      console.error(err);
    }
    let t1 = new Date();

    if (utils.dateDiff(t0, t1, 'seconds') > (dynamodb.getExecutionMaxTime() - 80)) {
      break;
    }
  }

  /*  If results is not empty, more procesing is required
       re-invoke lambda with remaining symbolGroups */
  if (symbolGroups.length) {
    let reinvokeEvent = {
      'symbolGroups': symbolGroups,
      'compDate': endDate,
      'processStats': processStats,
    };

    let description = `Continuing checkForAdjustments. ` +
      `${symbolGroups.length} symbol groups still to complete`;

    await invokeLambda('checkForAdjustments', reinvokeEvent, description);
  } else {
    let description = `Completed checkForAdjustments. Executing reloadQuote`;
    await invokeLambda('reloadQuote', {}, description);

    await
    sns.publishMsg(snsArn, `${JSON.stringify(processStats, null, 2)}`,
      'checkForAdjustments completed.');
  }

  return true;
};

let invokeLambda = async function(lambdaName, event, description) {
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
  try {
    await lambda.invoke({
      FunctionName: lambdaName,
      InvocationType: 'Event',
      Payload: JSON.stringify(event, null, 2),
    }).promise();
    console.log(`Function ${lambdaName} executed with event: `,
      `${JSON.stringify(event)}`);
    return true;
  } catch (err) {
    throw err;
  }
};

module.exports = {
  checkForAdjustments: checkForAdjustments,
};

// dynamodb.setLocalAccessConfig();
// checkForAdjustments({
//   compDate: '2018-08-10',
//   processStats: {
//     symbolsChecked: 5,
//     symbolReloadRecordsCreated: 1,
//   },
// });
