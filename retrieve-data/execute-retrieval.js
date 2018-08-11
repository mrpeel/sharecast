'use strict';

const retrieval = require('./dynamo-retrieve-share-data');
const utils = require('./utils');

/**  Executes all retrieval and update logic for the day's data
 * @param {Object} lastExecutionDetails
 */
let executeAll = async function(lastExecutionDetails) {
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
      'executeCompanyQuoteRetrieval9',
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

    switch (execute) {
      case 0:
        console.log('Executing retrieve financial indicators');
        await retrieval.executeFinancialIndicators();
        break;

      case 1:
        console.log('Executing retrieve company metrics');
        await retrieval.executeCompanyMetrics();
        break;

      case 2:
        console.log('Executing retrieve index quotes');
        await retrieval.executeIndexQuoteRetrieval();
        break;

      case 3:
        console.log('Executing retrieve company quotes - phase 1');
        await retrieval.executeCompanyQuoteRetrieval({
          startRec: 0,
          endRec: 199,
        });
        break;

      case 4:
        console.log('Executing retrieve company quotes - phase 2');
        await retrieval.executeCompanyQuoteRetrieval({
          startRec: 200,
          endRec: 399,
        });
        break;

      case 5:
        console.log('Executing retrieve company quotes - phase 3');
        await retrieval.executeCompanyQuoteRetrieval({
          startRec: 400,
          endRec: 599,
        });
        break;

      case 6:
        console.log('Executing retrieve company quotes - phase 4');
        await retrieval.executeCompanyQuoteRetrieval({
          startRec: 600,
          endRec: 799,
        });
        break;

      case 7:
        console.log('Executing retrieve company quotes - phase 5');
        await retrieval.executeCompanyQuoteRetrieval({
          startRec: 800,
          endRec: 999,
        });
        break;

      case 8:
        console.log('Executing retrieve company quotes - phase 6');
        await retrieval.executeCompanyQuoteRetrieval({
          startRec: 1000,
          endRec: 1199,
        });
        break;

      case 9:
        console.log('Executing retrieve company quotes - phase 7');
        await retrieval.executeCompanyQuoteRetrieval({
          startRec: 1200,
          endRec: 1399,
        });
        break;

      case 10:
        console.log('Executing retrieve company quotes - phase 8');
        await retrieval.executeCompanyQuoteRetrieval({
          startRec: 1400,
          endRec: 1599,
        });
        break;

      case 11:
        console.log('Executing retrieve company quotes - phase 9');
        await retrieval.executeCompanyQuoteRetrieval({
          startRec: 1600,
        });
        break;

      case 12:
        console.log('Executing update company quotes with metrics - phase 1');
        await retrieval.executeMetricsUpdate({
          startRec: 0,
          endRec: 599,
        });
        break;

      case 13:
        console.log('Executing update company quotes with metrics - phase 2');
        await retrieval.executeMetricsUpdate({
          startRec: 600,
          endRec: 1199,
        });
        break;

      case 14:
        console.log('Executing update company quotes with metrics - phase 3');
        await retrieval.executeMetricsUpdate({
          startRec: 1200,
          startRec: 1799,
        });
        break;

      case 15:
        console.log('Executing update company quotes with metrics - phase 4');
        await retrieval.executeMetricsUpdate({
          startRec: 1800,
        });
        break;
    }

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
};
