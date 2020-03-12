'use strict';

const retrieval = require('./retrieve/dynamo-retrieve-share-data');
const checkAdjustedPrices = require('./retrieve/check-adjusted-prices');
const retrieveAdjustedPrices = require('./retrieve/retrieve-adjusted-close-history');
const sns = require('./publish-sns');
const snsArn = 'arn:aws:sns:ap-southeast-2:815588223950:lambda-activity';

/* Individual handler */
let retrievalHandler = async function(event, context) {
  let functionName = event.retrievalFunction || '';
  console.log(`retrievalHandler called with event: ${JSON.stringify(event)}`);

  try {
    let t0 = getTiming();
    let executionResult;

    // Execute the specified function
    switch (functionName) {
      case 'executeFinancialIndicators':
        console.log('Executing retrieve financial indicators');
        executionResult = await retrieval.executeFinancialIndicators();
        break;

      case 'executeIndexQuoteRetrieval':
        console.log('Executing retrieve index quotes');
        executionResult = await retrieval.executeIndexQuoteRetrieval();
        break;

      case 'executeCompanyQuoteRetrieval1':
        console.log('Executing retrieve company quotes - phase 1');
        executionResult = await retrieval.executeCompanyQuoteRetrieval({
          startRec: 0,
          endRec: 599,
        });
        break;

      case 'executeCompanyQuoteRetrieval2':
        console.log('Executing retrieve company quotes - phase 2');
        executionResult = await retrieval.executeCompanyQuoteRetrieval({
          startRec: 600,
          endRec: 1199,
        });
        break;

      case 'executeCompanyQuoteRetrieval3':
        console.log('Executing retrieve company quotes - phase 3');
        executionResult = await retrieval.executeCompanyQuoteRetrieval({
          startRec: 1200,
          endRec: 1799,
        });
        break;

      case 'executeCompanyQuoteRetrieval4':
        console.log('Executing retrieve company quotes - phase 4');
        executionResult = await retrieval.executeCompanyQuoteRetrieval({
          startRec: 1800,
          endRec: 2399,
        });
        break;

      case 'executeCompanyQuoteRetrieval5':
        console.log('Executing retrieve company quotes - phase 5');
        executionResult = await retrieval.executeCompanyQuoteRetrieval({
          startRec: 2400,
        });
        break;
    }

    let duration = getTiming(t0);

    // Update the last executed function
    let logMessage = `${JSON.stringify(executionResult, null, 2)}`;
    let msgSubject = `Lambda ${functionName} completed in ${duration} seconds`;

    console.log(logMessage);

    // Send a confirmation email
    await sns.publishMsg(snsArn,
      logMessage,
      msgSubject);

    context.succeed();
  } catch (err) {
    console.error(`retrieveShareData function ${functionName} failed: `, err);
    try {
      await
      sns.publishMsg(snsArn,
        err,
        `Lambda retrieveShareData ${functionName} failed`);
    } catch (err) {}
    context.fail(`retrieveShareData ${functionName} failed`);
  }
};

let checkForAdjustmentsHandler = async function(event, context) {
  try {
    await checkAdjustedPrices.checkForAdjustments(event);
    context.succeed();
  } catch (err) {
    console.error('checkForAdjustments function failed: ', err);
    try {
      await
      sns.publishMsg(snsArn,
        err,
        'Lambda checkForAdjustments failed');
    } catch (err) {}
    context.fail('checkForAdjustments function failed');
  }
};

let reloadQuoteHandler = async function(event, context) {
  try {
    await retrieveAdjustedPrices.reloadQuote(event);
    context.succeed();
  } catch (err) {
    console.error('reloadQuote function failed: ', err);
    try {
      await sns.publishMsg(snsArn,
        err,
        'Lambda reloadQuote failed');
    } catch (err) {}
    context.fail('reloadQuote function failed');
  }
};

let getTiming = function(start) {
  if (!start) {
    return process.hrtime();
  }
  let end = process.hrtime(start);
  return end[0] + (end[1] / 1000000000);
};


module.exports = {
  retrievalHandler: retrievalHandler,
  checkForAdjustmentsHandler: checkForAdjustmentsHandler,
  reloadQuote: reloadQuoteHandler,
};

// let event = {
//   retrievalFunction: 'executeFinancialIndicators',
// };
// retrievalHandler(event, {
//   succeed: function() {
//     console.log('Succeeded');
//   },
//   fail: function(errMsg) {
//     console.log('Failed');
//   },
// });
