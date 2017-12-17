'use strict';

const lambdahandler = require('./index');

let context = {
  retrievalFunction: 'executeFinancialIndicators',
};
lambdahandler.retrievalHandler(null, context);


// 'executeCompanyMetrics',
// 'executeIndexQuoteRetrieval',
// 'executeCompanyQuoteRetrieval1',
// 'executeCompanyQuoteRetrieval2',
// 'executeCompanyQuoteRetrieval3',
// 'executeCompanyQuoteRetrieval4',
// 'executeCompanyQuoteRetrieval5',
// 'executeCompanyQuoteRetrieval6',
// 'executeCompanyQuoteRetrieval7',
// 'executeCompanyQuoteRetrieval8',
// 'executeMetricsUpdate1',
// 'executeMetricsUpdate2',
// 'executeMetricsUpdate3',
// 'executeMetricsUpdate4'
