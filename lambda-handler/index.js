'use strict';

const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const retrieval = require('./retrieve/dynamo-retrieve-share-data');
const checkAdjustedPrices = require('./retrieve/check-adjusted-prices');
const retrieveAdjustedPrices = require('./retrieve/retrieve-adjusted-close-history');
const processReturns = require('./retrieve/process-returns');
const utils = require('./retrieve/utils');
const sns = require('./publish-sns');
const aws = require('aws-sdk');

const lambda = new aws.Lambda({
  region: 'ap-southeast-2',
});

const snsArn = 'arn:aws:sns:ap-southeast-2:815588223950:lambda-activity';

let invokeLambda = function(lambdaName, event) {
  return new Promise(function(resolve, reject) {
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

/**  Executes function
*/
let dailyHandler = asyncify(function(event, context) {
  try {
    let lastExecuted;
    let executionList;
    let execute;
    let executionOrder = ['executeFinancialIndicators',
      'executeCompanyMetrics',
      'executeIndexQuoteRetrieval',
      'executeCompanyQuoteRetrieval1',
      'executeCompanyQuoteRetrieval2',
      'executeCompanyQuoteRetrieval3',
      'executeCompanyQuoteRetrieval4',
      'executeCompanyQuoteRetrieval5',
      'executeCompanyQuoteRetrieval6',
      'executeCompanyQuoteRetrieval7',
      'executeCompanyQuoteRetrieval8',
      'executeMetricsUpdate1',
      'executeMetricsUpdate2',
      'executeMetricsUpdate3',
      'executeMetricsUpdate4',
    ];

    // Check if there is a previously executed function in the chain
    if (event && event.lastExecuted) {
      lastExecuted = event.lastExecuted;
    }

    // Copy across the execution list if it exists
    if (event && event.executionList) {
      executionList = event.executionList;
    } else {
      executionList = [];
    }

    // Set-up the index for which share retrieval function to execute
    if (lastExecuted && executionOrder.indexOf(lastExecuted) === -1) {
      console.error('Unexpected value for lastExecuted function: ',
        lastExecuted);
      return;
    } else if (lastExecuted && executionOrder.indexOf(lastExecuted) > -1) {
      execute = executionOrder.indexOf(lastExecuted) + 1;
    } else {
      execute = 0;
    }

    let t0 = new Date();

    // Execute the specified function
    switch (executionOrder[execute]) {
      case 'executeFinancialIndicators':
        console.log('Executing retrieve financial indicators');
        awaitify(retrieval.executeFinancialIndicators());
        break;

      case 'executeCompanyMetrics':
        console.log('Executing retrieve company metrics');
        awaitify(retrieval.executeCompanyMetrics());
        break;

      case 'executeIndexQuoteRetrieval':
        console.log('Executing retrieve index quotes');
        awaitify(retrieval.executeIndexQuoteRetrieval());
        break;

      case 'executeCompanyQuoteRetrieval1':
        console.log('Executing retrieve company quotes - phase 1');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 0,
          endRec: 249,
        }));
        break;

      case 'executeCompanyQuoteRetrieval2':
        console.log('Executing retrieve company quotes - phase 2');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 250,
          endRec: 499,
        }));
        break;

      case 'executeCompanyQuoteRetrieval3':
        console.log('Executing retrieve company quotes - phase 3');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 500,
          endRec: 749,
        }));
        break;

      case 'executeCompanyQuoteRetrieval4':
        console.log('Executing retrieve company quotes - phase 4');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 750,
          endRec: 999,
        }));
        break;

      case 'executeCompanyQuoteRetrieval5':
        console.log('Executing retrieve company quotes - phase 5');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 1000,
          endRec: 1249,
        }));
        break;

      case 'executeCompanyQuoteRetrieval6':
        console.log('Executing retrieve company quotes - phase 6');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 1250,
          endRec: 1499,
        }));
        break;

      case 'executeCompanyQuoteRetrieval7':
        console.log('Executing retrieve company quotes - phase 7');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 1500,
          endRec: 1749,
        }));
        break;

      case 'executeCompanyQuoteRetrieval8':
        console.log('Executing retrieve company quotes - phase 8');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 1750,
        }));
        break;


      case 'executeMetricsUpdate1':
        console.log('Executing update company quotes with metrics - phase 1');
        awaitify(retrieval.executeMetricsUpdate({
          startRec: 0,
          endRec: 599,
        }));
        break;

      case 'executeMetricsUpdate2':
        console.log('Executing update company quotes with metrics - phase 2');
        awaitify(retrieval.executeMetricsUpdate({
          startRec: 600,
          endRec: 1199,
        }));
        break;

      case 'executeMetricsUpdate3':
        console.log('Executing update company quotes with metrics - phase 3');
        awaitify(retrieval.executeMetricsUpdate({
          startRec: 1200,
          endRec: 1799,
        }));
        break;

      case 'executeMetricsUpdate4':
        console.log('Executing update company quotes with metrics - phase 4');
        awaitify(retrieval.executeMetricsUpdate({
          startRec: 1800,
        }));
        break;
    }

    let t1 = new Date();

    // Update the last executed function
    lastExecuted = executionOrder[execute];

    // Update the last executed function
    let logMessage = executionOrder[execute] + ' took ' +
      utils.dateDiff(t0, t1, 'seconds') + ' seconds to execute.';
    let msgSubject = 'Lambda ' + executionOrder[execute] + ' completed';

    console.log(logMessage);

    // Send a confirmation email
    awaitify(
      sns.publishMsg(snsArn,
        logMessage,
        msgSubject));


    // Add completed function to execution list
    executionList.push(logMessage);

    // Check whether there are more functions in the chain to execute
    if (execute >= (executionOrder.length - 1)) {
      // Finished the last item, log what happened
      console.log('--------- All done --------');
      executionList.forEach((execution) => {
        console.log(execution);
      });

      context.succeed(event.executeFunction);
    } else {
      // More functions to execute, so call same lambda with updated details
      awaitify(invokeLambda('retrieveShareData',
        {
          lastExecuted: lastExecuted,
          executionList: executionList,
        }));
      context.succeed(event.executeFunction);
    }
  } catch (err) {
    console.error('retrieveShareData function failed: ', err);
    try {
      awaitify(
        sns.publishMsg(snsArn,
          err,
          'Lambda retrieveShareData failed'));
    } catch (err) {}
    context.fail('retrieveShareData function failed');
  }
});

let checkForAdjustmentsHandler = asyncify(function(event, context) {
  try {
    awaitify(checkAdjustedPrices.checkForAdjustments(event));
    context.succeed();
  } catch (err) {
    console.error('checkForAdjustments function failed: ', err);
    try {
      awaitify(
        sns.publishMsg(snsArn,
          err,
          'Lambda checkForAdjustments failed'));
    } catch (err) {}
    context.fail('checkForAdjustments function failed');
  }
});

let processReturnsHandler = asyncify(function(event, context) {
  try {
    awaitify(processReturns.processReturns(event));
    context.succeed();
  } catch (err) {
    console.error('processReturns function failed: ', err);
    try {
      awaitify(
        sns.publishMsg(snsArn,
          err,
          'Lambda processReturns failed'));
    } catch (err) {}
    context.fail('processReturns function failed');
  }
});

let reloadQuoteHandler = asyncify(function(event, context) {
  try {
    awaitify(retrieveAdjustedPrices.reloadQuote(event));
    context.succeed();
  } catch (err) {
    console.error('reloadQuote function failed: ', err);
    try {
      awaitify(
        sns.publishMsg(snsArn,
          err,
          'Lambda reloadQuote failed'));
    } catch (err) {}
    context.fail('reloadQuote function failed');
  }
});


module.exports = {
  dailyHandler: dailyHandler,
  checkForAdjustmentsHandler: checkForAdjustmentsHandler,
  processReturnsHandler: processReturnsHandler,
  reloadQuote: reloadQuoteHandler,
};
