'use strict';

const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const retrieval = require('./retrieve/dynamo-retrieve-share-data');
const utils = require('./retrieve/utils');

/**  Executes function
*/
let handler = asyncify(function(event, context) {
  try {
    let t0 = new Date();

    switch (event.executeFunction) {
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
          endRec: 599,
        }));
        break;

      case 'executeCompanyQuoteRetrieval2':
        console.log('Executing retrieve company quotes - phase 2');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 600,
          endRec: 1199,
        }));
        break;

      case 'executeCompanyQuoteRetrieval3':
        console.log('Executing retrieve company quotes - phase 3');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 1200,
          endRec: 1799,
        }));
        break;

      case 'executeCompanyQuoteRetrieval4':
        console.log('Executing retrieve company quotes - phase 4');
        awaitify(retrieval.executeCompanyQuoteRetrieval({
          startRec: 1800,
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
          startRec: 1799,
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

    console.log(event.executeFunction + ' took ' +
      utils.dateDiff(t0, t1, 'seconds') + ' seconds to execute.');

    context.succeed(event.executeFunction);
  } catch (err) {
    console.error(event.executeFunction, ' function failed: ', err);
    context.fail(event.executeFunction + ' function failed: ');
  }
});

module.exports.handler = handler;
