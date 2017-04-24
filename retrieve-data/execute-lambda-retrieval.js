'use strict';

const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const utils = require('./utils');
const aws = require('aws-sdk');
const lambda = new aws.Lambda({
  region: 'ap-southeast-2',
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

/**  Executes all retrieval and update logic for the day's data
*/
let executeAll = asyncify(function(lastExecutionDetails) {
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
      'executeMetricsUpdate1',
      'executeMetricsUpdate2',
      'executeMetricsUpdate3',
      'executeMetricsUpdate4',
    ];

    if (lastExecutionDetails && lastExecutionDetails.lastExecuted) {
      lastExecuted = lastExecutionDetails.lastExecuted;
    }

    if (lastExecutionDetails && lastExecutionDetails.executionList) {
      executionList = lastExecutionDetails.executionList;
    } else {
      executionList = [];
    }


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

    console.log('Executing ', executionOrder[execute]);

    awaitify(invokeLambda('retrieveShareData', {
      executeFunction: executionOrder[execute],
    }));

    let t1 = new Date();

    lastExecuted = executionOrder[execute];

    executionList.push(lastExecuted + ' took ' +
      utils.dateDiff(t0, t1, 'seconds') + ' seconds to execute.');

    if (execute >= (executionOrder.length - 1)) {
      // finished the last item, log what happened
      console.log('--------- All done --------');
      executionList.forEach((execution) => {
        console.log(execution);
      });
    } else {
      // Re-call same function with updated details
      executeAll({
        lastExecuted: lastExecuted,
        executionList: executionList,
      });
    }
  } catch (err) {
    console.error('executeAll function failed: ', err);
  }
});


executeAll();
