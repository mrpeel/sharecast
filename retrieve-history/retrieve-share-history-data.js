const yahooFinance = require('yahoo-finance');
const utils = require('../retrieve-data/utils');
const dynamodb = require('../retrieve-data/dynamodb');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const shareRetrieve = require('../retrieve-data/dynamo-retrieve-share-data');
const metricsRetrieve = require('../retrieve-data/dynamo-retrieve-google-company-data');
const fiRetrieve = require('../retrieve-data/dynamo-retrieve-financial-indicator-data');
const processRollups = require('../retrieve-data/process-rollups');
const histSymbols = require('./symbols.json');
const lastCompany = require('./last-company.json');
const moment = require('moment-timezone');
const sns = require('./publish-sns');
const snsArn = 'arn:aws:sns:ap-southeast-2:815588223950:lambda-activity';

let indexValsLookup = {};
let fiValsLookup = {};
let prepValues = [];
let indices = [];
let financiaIndicators = [];
let metrics = [];
let insert = [];
let historyReference = {};
let dividends = {};
let bollingerLastValues = {};
let lastCompanyVal = '';


let setupHistorySymbols = function() {
  try {
    console.log('----- Start setup symbols -----');
    let indexValues = histSymbols.indices;
    let wSymbolLookup = {};
    let wIndexLookup = {};
    let wIndexSymbols = [];
    let wCompanyLookup = {};
    let wIndices = [];
    let wCompanies = [];
    let returnVal = {};


    indexValues.forEach((indexValue) => {
      console.log(indexValue);
      wIndexLookup[indexValue['yahoo-symbol']] = indexValue['symbol'];
      wSymbolLookup[indexValue['yahoo-symbol']] = indexValue['symbol'];
      wIndexSymbols.push(indexValue['symbol']);
    });

    wIndices = utils.createFieldArray(wIndexLookup);

    returnVal = {
      indexLookup: wIndexLookup,
      indices: wIndices,
      symbolLookup: wSymbolLookup,
      indexSymbols: wIndexSymbols,
    };

    let companyValues = histSymbols.companies;

    companyValues.forEach((companyValue) => {
      wCompanyLookup[companyValue['yahoo-symbol']] = companyValue['symbol'];
      wSymbolLookup[companyValue['yahoo-symbol']] = companyValue['symbol'];
    });

    wCompanies = utils.createFieldArray(wCompanyLookup);

    returnVal.companyLookup = wCompanyLookup;
    returnVal.companies = wCompanies;

    return returnVal;
  } catch (err) {
    console.log(err);
  }
};

let retrieveHistory = function(symbol, startDate, endDate, interval) {
  return new Promise(function(resolve, reject) {
    let historyOptions = {
      from: startDate,
      to: endDate,
      period: interval,
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

/* let retrieveDividendHistory = function(symbol, startDate, endDate) {
  return new Promise(function(resolve, reject) {
    let historyOptions = {
      from: startDate,
      to: endDate,
      period: 'v',
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
};*/


/* let processIndexHistoryResults = asyncfunction(results) {
  if (results) {
    Object.keys(results).forEach(function(symbolResults) {
      // Multiple  symbols returned
      results[symbolResults].forEach(function(indResult) {
        awaitprocessIndexHistoryResult(indResult));
      });
    });
  }
}); */

/* let processIndexHistoryResult = asyncfunction(result) {
  result.lastTradeDate = utils.returnDateAsString(result.date);
  // Convert yahoo symbol to generic symbol
  result.symbol = symbolLookup[result.symbol];

  // Remove oriingal date field
  delete result['date'];

  Object.keys(result).forEach(function(field) {
    // Reset number here required
    result[field] = utils.checkForNumber(result[field]);
  });
  // Insert result in dynamodb
  awaitshareRetrieve.writeIndexQuote(result));
}); */

/*
setupSymbols();

let today = new Date();

let dateString = today.getFullYear() + '-' +
('0' + (today.getMonth() + 1)).slice(-2) + '-' +
('0' + today.getDate()).slice(-2);


// Retrieve dividend history
for (companyCounter = 0; companyCounter < companies.length;
  companyCounter += 10) {
  // symbolGroups.push(companies.slice(companyCounter, companyCounter + 10));
  let companySymbols = companies.slice(companyCounter, companyCounter + 10);
  let internalCounter = companyCounter;

  if (!utils.doesDataFileExist('companies-dividend-history-' + internalCounter
      + '-' + dateString + '.csv')) {
    setTimeout(function() {
      console.log('Processing companyCounter: ' + internalCounter);
      retrieveDividendHistory(companySymbols, '2012-01-01', '2017-01-29')
        .then(function(results) {
          // Reset fields for companies
          csvFields = [];
          csvData = [];

          processHistoryResults(results);

          if (csvData.length > 0) {
            utils.writeToCsv(csvData, csvFields, 'companies-dividend-history-' +
              internalCounter);
          } else {
            console.log('No history data to save');
          }
        }).catch(function(err) {
        console.log(err);
      });
    }, internalCounter * 20);
  } else {
    console.log('Skipping companyCounter: ' + internalCounter);
  }
} */


/* let getDividendHistory = asyncfunction() {
  let symbolResult = awaitshareRetrieve.setupSymbols());

  let symbolLookup = symbolResult.symbolLookup;
  let indexLookup = symbolResult.indexLookup;
  let indexSymbols = symbolResult.indexSymbols;
  let companyLookup = symbolResult.companyLookup;
  let indices = symbolResult.indices;
  let companies = symbolResult.companies;

  // Split companies into groups of 10 so each request contains 10
  for (let companyCounter = 0; companyCounter < companies.length;
    companyCounter += 10) {
    // symbolGroups.push(companies.slice(companyCounter, companyCounter + 10));
    let companySymbols = companies.slice(companyCounter, companyCounter + 10);
    let internalCounter = companyCounter;

    let insertDetails = {
      tableName: 'companyMetrics',
      primaryKey: ['symbol', 'metricsDate'],
    };


    console.log('Processing companyCounter: ' + internalCounter);
    // awaitretrieveHistory(companySymbols,
    //   '2006-01-01', '2017-01-29', 'd')
    try {
      let results = awaitretrieveDividendHistory(companySymbols,
        '2006-01-01', '2017-01-29'));
      // Reset fields for companies
      let data = [];

      processHistoryResults(results);

      if (data.length > 0) {
        data.forEach((dividendRecord) => {
          dividendRecord['DividendPerShare'] = dividendRecord['dividends'];
          dividendRecord['metricsDate'] = dividendRecord['date'];
          dividendRecord['yearMonth'] = dividendRecord['metricsDate']
            .substring(0, 7)
            .replace('-', '');

          delete dividendRecord['dividends'];
          delete dividendRecord['date'];

          // console.log(dividendRecord);
          insertDetails.values = dividendRecord;

          awaitdynamodb.insertRecord(insertDetails));
        });
      } else {
        console.log('No history data to save');
      }
    } catch (err) {
      console.log(err);
    }
  }
}); */

/* retrieveHistory(indices, '2012-01-01',
  '2017-01-29', 'd')
  .then(function(results) {
    return processHistoryResults(results);
  })
  .then(function() {
    if (csvData.length > 0) {
      utils.writeToCsv(csvData, csvFields, 'indice-history');
    } else {
      console.log('No index history data to save');
    }

    return true;
  }).then(function() {

// Split companies into groups of 10 so each request contains 10
for (companyCounter = 0; companyCounter < companies.length;
  companyCounter += 10) {
  // symbolGroups.push(companies.slice(companyCounter, companyCounter + 10));
  let companySymbols = companies.slice(companyCounter, companyCounter + 10);
  let internalCounter = companyCounter;

  setTimeout(function() {
    console.log('companyCounter: ' + internalCounter);
    retrieveHistory(companySymbols, '2012-01-01',
      '2017-01-29', 'd')
      .then(function(results) {
        // Reset fields for companies
        csvFields = [];
        csvData = [];

        processHistoryResults(results);

        if (csvData.length > 0) {
          utils.writeToCsv(csvData, csvFields, 'companies-history-' +
            internalCounter);
        } else {
          console.log('No history data to save');
        }
      }).catch(function(err) {
      console.log(err);
    });
  }, internalCounter * 100);
}
/* })
  .catch(function(err) {
    console.log(err);
  }); */

/* let getIndexHistory = asyncfunction() {
  let symbolResult = awaitshareRetrieve.setupSymbols());

  let symbolLookup = symbolResult.symbolLookup;
  let indexLookup = symbolResult.indexLookup;
  let indexSymbols = symbolResult.indexSymbols;
  let companyLookup = symbolResult.companyLookup;
  let indices = symbolResult.indices;
  let companies = symbolResult.companies;

  let results = awaitretrieveHistory(indices,
    '2006-07-01', '2011-12-31', 'd'));

  processIndexHistoryResults(results);
}); */

let get52WeekResults = asyncfunction(symbolGroup, symbolLookup,
  referenceDate) {
  let historyResults = {};
  let endDate = utils.dateAdd(referenceDate, 'days', -1);
  let startDate = utils.dateAdd(referenceDate, 'years', -1);

  let results = awaitretrieveHistory(symbolGroup,
    startDate, endDate, 'd'));

  /* let queryDetails = {
    tableName: 'companyQuotes',
    keyConditionExpression: 'symbol = :symbol and ' +
      'quoteDate between :startDate and :endDate',
    filterExpression: 'attribute_exists(adjustedPrice)',
    expressionAttributeValues: {
      ':startDate': startDate,
      ':endDate': endDate,
    },
    projectionExpression: 'symbol, quoteDate, adjustedPrice',
  };

  symbolGroup.forEach((symbol) => {
    queryDetails.expressionAttributeValues[':symbol'] = symbolLookup[symbol];
    let queryResults = awaitdynamodb.queryTable(queryDetails));
    queryResults.forEach((result) => {
      let symbol = result.symbol;
      let adjustedPrice = result.adjustedPrice;
      let quoteDate = result.quoteDate;

      if (!historyResults[symbol]) {
        historyResults[symbol] = {};
      }
      historyResults[symbol][quoteDate] = adjustedPrice;
    });
  }); */

  if (results) {
    Object.keys(results).forEach(function(symbolResults) {
      // Check if mult diensional
      let currentResults = results[symbolResults];
      if (Array.isArray(currentResults)) {
        currentResults.forEach((indResult) => {
          if (indResult.adjClose) {
            let symbol = symbolLookup[indResult.symbol];
            let quoteDate = utils.returnDateAsString(indResult.date);
            if (!historyResults[symbol]) {
              historyResults[symbol] = {};
            }
            historyResults[symbol][quoteDate] = indResult.adjClose;
          }
        });
      }
    });
  }

  return historyResults;
});

let setLastCompany = function(companySymbol) {
  if (companySymbol !== lastCompanyVal) {
    let companyObject = {
      'lastCompanySymbol': companySymbol,
    };
    utils.writeJSONfile(companyObject, './last-company.json');
    lastCompanyVal = companySymbol;
  }
};


let getCompanyHistory = asyncfunction() {
  let currentCompany;
  try {
    dynamodb.setLocalAccessConfig();
    // override console within this block
    let symbolResult = setupHistorySymbols();

    let symbolLookup = symbolResult.symbolLookup;
    let companies = symbolResult.companies;
    let symbolGroups = [];
    let filteredCompanies = [];

    lastCompanyVal = lastCompany.lastCompanySymbol;

    // Work through companies one by one and retrieve values
    companies.forEach((companySymbol) => {
      // Skip previous companies
      if (companySymbol >= lastCompanyVal) {
        filteredCompanies.push(companySymbol);
      }
    });


    for (let companyCounter = 0; companyCounter < filteredCompanies.length;
      companyCounter += 1) {
      symbolGroups.push(filteredCompanies.slice(companyCounter,
        companyCounter + 1));
    }

    symbolGroups.forEach((symbolGroup) => {
      let t0 = new Date();
      let results = awaitretrieveHistory(symbolGroup,
        '2007-07-01', '2017-02-03', 'd'));
      let t1 = new Date();
      console.warn('Retrieve data took ' +
        utils.dateDiff(t0, t1, 'seconds') + ' seconds.');

      prepValues = [];
      indices = [];
      financiaIndicators = [];
      metrics = [];
      insert = [];
      bollingerLastValues = {};

      historyReference = awaitget52WeekResults(symbolGroup, symbolLookup,
        '2007-07-01'));

      dividends = awaitgetDividendsforDate(symbolGroup, symbolLookup,
        '2006-07-01', '2017-02-03'));

      awaitprocessCompanyHistoryResults(results, symbolLookup));

    /* let t2 = new Date();
    console.warn('Process history results took ' +
      utils.dateDiff(t1, t2, 'seconds') + ' seconds.');

    console.warn('Prep values sum: ', stats.sum(prepValues),
      ' avg: ', stats.average(prepValues),
      ', max: ', stats.max(prepValues));
    console.warn('Add index data sum: ', stats.sum(indices),
      ' avg: ', stats.average(indices),
      ', max: ', stats.max(indices));
    console.warn('Add financial indicator data sum: ',
      stats.sum(financiaIndicators),
      'avg: ', stats.average(financiaIndicators), ', max: ',
      stats.max(financiaIndicators));
    console.warn('Add metrics data sum: ', stats.sum(metrics),
      'avg: ', stats.average(metrics),
      ', max: ', stats.max(metrics));
    console.warn('Insert data sum: ', stats.sum(insert),
      'avg: ', stats.average(insert),
      ', max: ', stats.max(insert)); */
    });
  } catch (err) {
    try {
      await
        sns.publishMsg(snsArn,
          'getCompanyHistory failed.  Error: ' + JSON.stringify(err),
          'getCompanyHistory failed'));
    } catch (err) {}

    console.error('Failed while processing: ', currentCompany);
    console.error(err);
  }
});

let processCompanyHistoryResults = asyncfunction(results, symbolLookup) {
  if (results) {
    /*    // Check if multi-dimensional
        if (Array.isArray(results)) {
          // Multiple  symbols returned
          results.forEach((indResult) => {
            Object.keys(indResult).forEach(function(symbolResults) {
              if (indResult[symbolResults].close) {
                awaitprocessCompanyHistoryResult(indResult[symbolResults],
                  symbolLookup));
              }
            });
          });
        } else {*/
    Object.keys(results).forEach(function(symbolResults) {
      // Check if mult diensional
      let currentResults = results[symbolResults];
      if (Array.isArray(currentResults)) {
        currentResults.forEach((indResult) => {
          if (indResult.close) {
            let timings = awaitprocessCompanyHistoryResult(indResult,
              symbolLookup));

            prepValues.push(timings.prepValues);
            indices.push(timings.indices);
            financiaIndicators.push(timings.financiaIndicators);
            metrics.push(timings.metrics);
            insert.push(timings.insert);
          }
        });
      } else {
        /* Single symbol returned - check whether this is a normal record,
            i.e. has Close val */
        if (results[symbolResults].close) {
          awaitprocessCompanyHistoryResult(results[symbolResults],
            symbolLookup));
        }
      }
    });
  }
// }
});

let processCompanyHistoryResult = asyncfunction(result, symbolLookup) {
  // overwrite standard console log
  // console.log = function() {};
  let t0 = new Date();
  let timings = {};

  // Check whether we are trying to process during an exclusion time
  if (checkExclusionTime('17:28', '17:48') || checkExclusionTime('23:58', '00:08')) {
    // Sleep for 5 minutes if in exclusion zone
    console.log(moment().tz('Australia/Sydney').format(),
      ': pausing for five minutes');
    awaitutils.sleep(5 * 60 * 1000));
  }

  // Set the current company being worked on
  setLastCompany(result.symbol);

  let ignoreVals = ['created', 'yearMonth', 'valueDate', 'metricsDate',
    'quoteDate'];
  result.lastTradeDate = utils.returnDateAsString(result.date);
  // Convert yahoo symbol to generic symbol
  result.symbol = symbolLookup[result.symbol];
  result.adjustedPrice = result.adjClose;
  result.previousClose = result.open;
  result.lastTradePriceOnly = result.close;
  result.daysHigh = result.high;
  result.daysLow = result.low;
  result.change = result.close - result.open;
  if (result.open === 0) {
    result.changeInPercent = 0;
  } else {
    result.changeInPercent = (result.close - result.open) / result.open || 1;
  }

  console.log(moment().tz('Australia/Sydney').format(), ': processing ',
    result.symbol, ', ', result.lastTradeDate);

  // Remove oriingal fields
  delete result['date'];
  delete result['adjClose'];
  delete result['open'];
  delete result['close'];
  delete result['high'];
  delete result['low'];

  Object.keys(result).forEach(function(field) {
    // Reset number here required
    result[field] = utils.checkForNumber(result[field]);
  });

  // Check we have history records for this symbol
  if (!historyReference[result.symbol]) {
    historyReference[result.symbol] = {};
  }

  // Add the current price to the history reference set
  historyReference[result.symbol][result.lastTradeDate] = result.adjustedPrice;

  // Calculate std deviation for 1, 2, 4, 8, 12, 26, 52 weeks
  let weekStats = processRollups.getWeeklyStats(
    historyReference[result.symbol],
    result.lastTradeDate);

  result['1WeekVolatility'] = weekStats['1WeekStdDev'];
  result['2WeekVolatility'] = weekStats['2WeekStdDev'];
  result['4WeekVolatility'] = weekStats['4WeekStdDev'];
  result['8WeekVolatility'] = weekStats['8WeekStdDev'];
  result['12WeekVolatility'] = weekStats['12WeekStdDev'];
  result['26WeekVolatility'] = weekStats['26WeekStdDev'];
  result['52WeekVolatility'] = weekStats['52WeekStdDev'];

  // Calculate 52 week high, low, change and percentChange
  result['52WeekHigh'] = weekStats['52WeekHigh'];
  result['52WeekLow'] = weekStats['52WeekLow'];
  result['changeFrom52WeekHigh'] = result.adjustedPrice -
  weekStats['52WeekHigh'];
  result['changeFrom52WeekLow'] = result.adjustedPrice -
  weekStats['52WeekLow'];
  if (weekStats['52WeekHigh']) {
    result['percebtChangeFrom52WeekHigh'] = (result.adjustedPrice -
    weekStats['52WeekHigh']) /
    weekStats['52WeekHigh'];
  } else {
    result['percebtChangeFrom52WeekHigh'] = 0;
  }
  if (weekStats['52WeekLow']) {
    result['percentChangeFrom52WeekLow'] = (result.adjustedPrice -
    weekStats['52WeekLow']) /
    weekStats['52WeekLow'];
  } else {
    result['percentChangeFrom52WeekLow'] = 0;
  }

  result['4WeekBollingerBandUpper'] = weekStats['4WeekBollingerBandUpper'];
  result['4WeekBollingerBandLower'] = weekStats['4WeekBollingerBandLower'];
  result['4WeekBollingerPrediction'] = 'Steady';
  if (result.adjustedPrice > weekStats['4WeekBollingerBandUpper']) {
    result['4WeekBollingerType'] = 'Above';
  } else if (result.adjustedPrice < weekStats['4WeekBollingerBandLower']) {
    result['4WeekBollingerType'] = 'Below';
  } else {
    result['4WeekBollingerType'] = 'Within';
  }


  result['12WeekBollingerBandUpper'] = weekStats['12WeekBollingerBandUpper'];
  result['12WeekBollingerBandLower'] = weekStats['12WeekBollingerBandLower'];
  result['12WeekBollingerPrediction'] = 'Steady';
  if (result.adjustedPrice > weekStats['12WeekBollingerBandUpper']) {
    result['12WeekBollingerType'] = 'Above';
  } else if (result.adjustedPrice < weekStats['12WeekBollingerBandLower']) {
    result['12WeekBollingerType'] = 'Below';
  } else {
    result['12WeekBollingerType'] = 'Within';
  }


  // Check for movements down from above upper band and up from below lower band
  if (bollingerLastValues[result.symbol]) {
    let last4Week = bollingerLastValues[result.symbol]['4WeekBollingerType'];
    let last12Week = bollingerLastValues[result.symbol]['12WeekBollingerType'];

    if (last4Week === 'Above' && result['4WeekBollingerType'] === 'Within') {
      result['4WeekBollingerPrediction'] = 'Falling';
    } else if (last4Week === 'Below' &&
      result['4WeekBollingerType'] === 'Within') {
      result['4WeekBollingerPrediction'] = 'Rising';
    }

    if (last12Week === 'Above' && result['12WeekBollingerType'] === 'Within') {
      result['12WeekBollingerPrediction'] = 'Falling';
    } else if (last12Week === 'Below' &&
      result['12WeekBollingerType'] === 'Within') {
      result['12WeekBollingerPrediction'] = 'Rising';
    }
  }

  bollingerLastValues[result.symbol] = {
    '4WeekBollingerType': result['4WeekBollingerType'],
    '12WeekBollingerType': result['12WeekBollingerType'],
  };

  // Check whether dividends exist for this symbol
  if (dividends[result.symbol]) {
    let maxDate = result.lastTradeDate;
    let dividendDate = '';
    Object.keys(dividends[result.symbol]).forEach((exDividendDate) => {
      if (exDividendDate > dividendDate && exDividendDate <= maxDate) {
        dividendDate = exDividendDate;
      }
    });

    // If an ex dividend date was found, add it to the record
    if (dividendDate !== '') {
      result['exDividendDate'] = dividendDate;
      result['exDividendPayout'] = dividends[result.symbol][dividendDate];
    }
  }


  let t1 = new Date();
  timings.prepValues = utils.dateDiff(t0, t1, 'milliseconds');

  // Get index values for date
  if (!indexValsLookup[result.lastTradeDate]) {
    let indexValRetrieve = await
      shareRetrieve.returnIndexDataForDate(result.lastTradeDate));
    indexValsLookup[result.lastTradeDate] = indexValRetrieve;
  }

  let indexVals = indexValsLookup[result.lastTradeDate];

  Object.keys(indexVals).forEach((indexKey) => {
    // Check if value should be ignored.  If not add to quoteVal
    if (ignoreVals.indexOf(indexKey) === -1 &&
      indexVals[indexKey] !== undefined &&
      indexVals[indexKey] !== null) {
      result[indexKey] = indexVals[indexKey];
    }
  });

  let t2 = new Date();
  timings.indices = utils.dateDiff(t1, t2, 'milliseconds');

  // Get financial indicator values for date
  if (!fiValsLookup[result.lastTradeDate]) {
    let fiValsRetrieve = await
      fiRetrieve.returnIndicatorValuesForDate(result.lastTradeDate));
    fiValsLookup[result.lastTradeDate] = fiValsRetrieve;
  }

  let fiVals = fiValsLookup[result.lastTradeDate];

  Object.keys(fiVals).forEach((fiKey) => {
    // Check if value should be ignored.  If not add to quoteVal
    if (ignoreVals.indexOf(fiKey) === -1 &&
      fiVals[fiKey] !== undefined &&
      fiVals[fiKey] !== null) {
      result[fiKey] = fiVals[fiKey];
    }
  });

  let t3 = new Date();
  timings.financiaIndicators = utils.dateDiff(t2, t3, 'milliseconds');


  // Get metric values for date
  let metricsVals = await
    metricsRetrieve.returnCompanyMetricValuesForDate(result.symbol,
      result.lastTradeDate));

  Object.keys(metricsVals).forEach((metricsKey) => {
    // Check if value should be ignored.  If not add to quoteVal
    if (ignoreVals.indexOf(metricsKey) === -1 &&
      metricsVals[metricsKey] !== undefined &&
      metricsVals[metricsKey] !== null) {
      result[metricsKey] = metricsVals[metricsKey];
    }
  });

  let t4 = new Date();
  timings.metrics = utils.dateDiff(t3, t4, 'milliseconds');

  // Insert result in dynamodb
  let insertResult = awaitwriteCompanyQuoteData(result));

  // Check whether it was inserted
  if (insertResult.result !== 'skipped') {
    /* Calculate and update total return and risk adjusted return
      for 1, 2, 4, 8, 12, 26, 52 weeks */
    awaitprocessRollups.updateReturns(
      result.symbol,
      result.quoteDate,
      result.adjustedPrice,
      historyReference[result.symbol],
      dividends[result.symbol] || {}));
  } else {
    console.log('Skipping update of previous records');
  }

  let t5 = new Date();
  timings.insert = utils.dateDiff(t4, t5, 'milliseconds');

  return timings;
// Add current quote to company adjusted price array
});

/**  Executes the retrieval and return of current company dividend
*      information for the date specified
* @param {Array} symbolGroup  one or more yahoo symbols to look up for dividend
* @param {String} retrievalDate - the date to retrieve dividends for
* @param {Array} symbolLookup - an array of key/value pairs to translate
*                               the symbols yahoo uses to the normal symbol
* @return {Object} an object for each company which has relevant dividend
*     information for the specified date in the form of:
*   {
*   'JBH': {
*           'exDividendDate': '2017-02-23',
*           'exDividendPayout': 1.02344,
*           },
*   }
*/
let getDividendsforDate = asyncfunction(symbolGroup, symbolLookup,
  startDate, endDate) {
  if (!symbolGroup) {
    console.error('getDividendsforDate error: no symbolGroup supplied');
    return;
  }

  if (!startDate) {
    console.error('getDividendsforDate error: no startDate supplied');
    return;
  }

  if (!endDate) {
    console.error('getDividendsforDate error: no startDate supplied');
    return;
  }

  if (!symbolLookup) {
    console.error('getDividendsforDate error: no symbolLookup supplied');
    return;
  }


  // Retrieve dividends
  // let dividendEndDate = utils.returnDateAsString(retrievalDate);
  // let dividendStartDate = utils.dateAdd(dividendEndDate, 'years', -2);
  // dividendStartDate = utils.dateAdd(dividendStartDate, 'days', 1);
  let dividendRetrieval = awaitshareRetrieve.retrieveDividends(symbolGroup,
    startDate, endDate));

  // Convert dividend results array to results object
  let dividendResults = {};

  // Single company result is returned as an array
  if (Array.isArray(dividendRetrieval) && dividendRetrieval.length > 0) {
    // Multiple  symbols returned
    dividendRetrieval.forEach((dividendRecord) => {
      processDividend(dividendRecord, symbolLookup, dividendResults);
    });
  } else {
    // Multi-company result is returned as an object
    Object.keys(dividendRetrieval).forEach((dividendCompany) => {
      dividendRetrieval[dividendCompany].forEach((dividendRecord) => {
        processDividend(dividendRecord, symbolLookup, dividendResults);
      });
    });
  }

  return dividendResults;

  /** Check if dividend record is the latest record for each symbol.  If it is
  *    pdate the recorded dividend
  * @param {Object} dividendRecord record to process
  * @param {Array} symbolLookup - an array of key/value pairs to translate
  *                               the symbols yahoo uses to the normal symbol
  */
  function processDividend(dividendRecord, symbolLookup) {
    let symbol = symbolLookup[dividendRecord.symbol];
    let exDividendDate = utils.returnDateAsString(dividendRecord['date']);
    let exDividendPayout = dividendRecord['dividends'];

    /* Check if symbol has already been recorded, or recorded with an earlier
      date */
    if (!dividendResults[symbol]) {
      dividendResults[symbol] = {};
    }

    dividendResults[symbol][exDividendDate] = exDividendPayout;
  }
});

/* companyQuote data
History
Date -> quoteDate, lastTradeDate
Open	-> open
High -> daysHigh
Low -> daysLow
Close	-> lastTradePriceOnly
AdjustedClose -> adjustedPrice
Volume-> volume


Current
{
  "52WeekHigh": 0.006,
  "52WeekLow": 0.002,
  "bookValue": 0.003,
  "change": 0.001,
  "changeInPercent": 0.5,
  "created": "2017-02-23T05:46:31+11:00",
  "daysHigh": 0.003,
  "daysLow": 0.003,
  "earningsPerShare": 0,
  "ebitda": -673892,
  "lastTradeDate": "30/1/17",
  "lastTradePriceOnly": 0.003,
  "marketCapitalization": 9070000,
  "open": 0.003,
  "pegRatio": 0,
  "previousClose": 0.002,
  "pricePerBook": 0.667,
  "pricePerSales": 6046156,
  "quoteDate": "2017-01-30",
  "shortRatio": 0,
  "stockExchange": "ASX",
  "symbol": "LNY",
  "volume": 4659784,
  "yearMonth": 201701
}
*/

/* Index history data

History:
open -> previousClose
high -> daysHigh
low -> daysLow
change -> open - close
changeInPercent -> change / open

{
  "adjClose": 4901.100098,
  "close": 4901.100098,
  "created": "2017-02-22T08:50:00+11:00",
  "high": 4930.399902,
  "lastTradeDate": "2011-02-24",
  "low": 4896.700195,
  "open": 4929.799805,
  "quoteDate": "2011-02-24",
  "symbol": "ALLORD",
  "volume": 1305370200,
  "yearMonth": "201102"
}

Current
{
  "52WeekHigh": {
    "N": "5880.9"
  },
  "52WeekLow": {
    "N": "4905.5"
  },
  "change": {
    "N": "-45.6"
  },
  "changeFrom52WeekHigh": {
    "N": "-94"
  },
  "changeFrom52WeekLow": {
    "N": "881.4"
  },
  "changeInPercent": {
    "N": "-0.0078000000000000005"
  },
  "created": {
    "S": "2017-02-24T20:29:25+11:00"
  },
  "daysHigh": {
    "N": "5832.5"
  },
  "daysLow": {
    "N": "5778.8"
  },
  "lastTradeDate": {
    "S": "2017-02-24"
  },
  "name": {
    "S": "ALL ORDINARIES"
  },
  "percebtChangeFrom52WeekHigh": {
    "N": "-0.016"
  },
  "percentChangeFrom52WeekLow": {
    "N": "0.1797"
  },
  "previousClose": {
    "N": "5832.5"
  },
  "quoteDate": {
    "S": "2017-02-24"
  },
  "symbol": {
    "S": "ALLORD"
  },
  "yearMonth": {
    "S": "201702"
  }
}
*/

/* let fixIndexHistory = asyncfunction() {
  let symbol = 'ASX';
  let queryDetails = {
    tableName: 'indexQuotes',
    keyConditionExpression: 'symbol = :symbol and ' +
      'quoteDate between :startDate and :endDate',
    expressionAttributeValues: {
      ':startDate': '2006-07-03',
      ':endDate': '2011-12-30',
      ':symbol': symbol,
    },
  };

  let updateDetails = {
    tableName: 'indexQuotes',
  };


  let indexQuotes = awaitdynamodb.queryTable(queryDetails));

  indexQuotes.forEach((indexQuote) => {
    // Set up the key: symbol and quoteDate
    updateDetails.key = {
      symbol: symbol,
      quoteDate: indexQuote['quoteDate'],
    };

    let updateExpression = 'set previousClose = :open, daysHigh = :high, ' +
      'daysLow = :low, change = :change, ' +
      'changeInPercent = :changeInPercent';

    let expressionAttributeValues = {
      ':open': indexQuote['open'],
      ':high': indexQuote['high'],
      ':low': indexQuote['low'],
      ':change': (indexQuote['open'] - indexQuote['close']),
      ':changeInPercent': ((indexQuote['open'] - indexQuote['close']) /
        indexQuote['open']),
    };

    updateDetails.updateExpression = updateExpression;
    updateDetails.expressionAttributeValues = expressionAttributeValues;

    try {
      awaitdynamodb.updateRecord(updateDetails));
    } catch (err) {
      console.log(err);
    }
  });
}); */

/* let addYearResultsToIndexHistory = asyncfunction() {
  let symbols = ['ALLORD', 'ASX'];

  symbols.forEach((symbol) => {
    let queryDetails = {
      tableName: 'indexQuotes',
      keyConditionExpression: 'symbol = :symbol and ' +
        'quoteDate between :startDate and :endDate',
      expressionAttributeValues: {
        ':startDate': '2007-07-01',
        ':endDate': '2017-01-29',
        ':symbol': symbol,
      },
    };

    let updateDetails = {
      tableName: 'indexQuotes',
    };


    let indexQuotes = awaitdynamodb.queryTable(queryDetails));

    indexQuotes.forEach((indexQuote) => {
      // Set up the key: symbol and quoteDate
      updateDetails.key = {
        symbol: symbol,
        quoteDate: indexQuote['quoteDate'],
      };

      let referenceValue = indexQuote['previousClose'] || 0;

      let startDate = utils.dateAdd(indexQuote['quoteDate'], 'weeks', -52);

      let query2Details = {
        tableName: 'indexQuotes',
        keyConditionExpression: 'symbol = :symbol and ' +
          'quoteDate between :startDate and :endDate',
        expressionAttributeValues: {
          ':startDate': startDate,
          ':endDate': indexQuote['quoteDate'],
          ':symbol': symbol,
        },
        projectionExpression: 'previousClose',
      };

      let quotesYear = awaitdynamodb.queryTable(query2Details));

      if (quotesYear.length) {
        let valArray = [];
        quotesYear.forEach((dayVal) => {
          valArray.push(dayVal['previousClose']);
        });

        let maxVal = stats.max(valArray);
        let minVal = stats.min(valArray);
        let changeMax = referenceValue - maxVal;
        let changeMin = referenceValue - minVal;
        let percentChangeMax = (referenceValue - maxVal) / maxVal;
        let percentChangeMin = (referenceValue - minVal) / minVal;

        let updateExpression = 'set #52WeekHigh = :52WeekHigh, ' +
          '#52WeekLow = :52WeekLow, ' +
          '#changeFrom52WeekHigh = :changeFrom52WeekHigh, ' +
          '#changeFrom52WeekLow = :changeFrom52WeekLow, ' +
          '#percebtChangeFrom52WeekHigh = :percebtChangeFrom52WeekHigh, ' +
          '#percentChangeFrom52WeekLow = :percentChangeFrom52WeekLow';

        let expressionAttributeNames = {
          '#52WeekHigh': '52WeekHigh',
          '#52WeekLow': '52WeekLow',
          '#changeFrom52WeekHigh': 'changeFrom52WeekHigh',
          '#changeFrom52WeekLow': 'changeFrom52WeekLow',
          '#percebtChangeFrom52WeekHigh': 'percebtChangeFrom52WeekHigh',
          '#percentChangeFrom52WeekLow': 'percentChangeFrom52WeekLow',
        };

        let expressionAttributeValues = {
          ':52WeekHigh': maxVal,
          ':52WeekLow': minVal,
          ':changeFrom52WeekHigh': changeMax,
          ':changeFrom52WeekLow': changeMin,
          ':percebtChangeFrom52WeekHigh': percentChangeMax,
          ':percentChangeFrom52WeekLow': percentChangeMin,
        };

        updateDetails.updateExpression = updateExpression;
        updateDetails.expressionAttributeValues = expressionAttributeValues;
        updateDetails.expressionAttributeNames = expressionAttributeNames;

        try {
          awaitdynamodb.updateRecord(updateDetails));
        } catch (err) {
          console.log(err);
        }
      }
    });
  });
}); */


/* let addDailyChangeToIndexHistory = asyncfunction() {
  let symbols = ['ALLORD', 'ASX'];

  symbols.forEach((symbol) => {
    let queryDetails = {
      tableName: 'indexQuotes',
      keyConditionExpression: 'symbol = :symbol and ' +
        'quoteDate between :startDate and :endDate',
      expressionAttributeValues: {
        ':startDate': '2007-07-01',
        ':endDate': '2017-01-29',
        ':symbol': symbol,
      },
      projectionExpression: 'symbol, quoteDate, previousClose',
    };

    let updateDetails = {
      tableName: 'indexQuotes',
    };


    let indexQuotes = awaitdynamodb.queryTable(queryDetails));

    indexQuotes.forEach((indexQuote) => {
      // Set up the key: symbol and quoteDate
      updateDetails.key = {
        symbol: symbol,
        quoteDate: indexQuote['quoteDate'],
      };

      let currentValue = indexQuote['previousClose'] || 0;

      let query2Details = {
        tableName: 'indexQuotes',
        keyConditionExpression: 'symbol = :symbol and ' +
          'quoteDate < :quoteDate',
        expressionAttributeValues: {
          ':quoteDate': indexQuote['quoteDate'],
          ':symbol': symbol,
        },
        reverseOrder: true,
        limit: 1,
        projectionExpression: 'previousClose',
      };

      let yesterdayQuote = awaitdynamodb.queryTable(query2Details));

      if (yesterdayQuote.length) {
        let previousValue = yesterdayQuote[0]['previousClose'];

        let change = currentValue - previousValue;
        let changeInPercent = (currentValue - previousValue) / previousValue;

        let updateExpression = 'set #change = :change, ' +
          '#changeInPercent = :changeInPercent';

        let expressionAttributeNames = {
          '#change': 'change',
          '#changeInPercent': 'changeInPercent',
        };

        let expressionAttributeValues = {
          ':change': change,
          ':changeInPercent': changeInPercent,
        };

        updateDetails.updateExpression = updateExpression;
        updateDetails.expressionAttributeValues = expressionAttributeValues;
        updateDetails.expressionAttributeNames = expressionAttributeNames;

        try {
          awaitdynamodb.updateRecord(updateDetails));
        } catch (err) {
          console.log(err);
        }
      }
    });
  });
}); */

/* let fixCompaniesList = asyncfunction() {
  let currentCompanySymbols = [];
  let companiesToRemove = [];
  let currentMetricsCompanies = [];
  let symbolResult = awaitshareRetrieve.setupSymbols());

  let symbolLookup = symbolResult.symbolLookup;
  let indexLookup = symbolResult.indexLookup;
  let indexSymbols = symbolResult.indexSymbols;
  let companyLookup = symbolResult.companyLookup;
  let indices = symbolResult.indices;
  let companies = symbolResult.companies;

  Object.keys(companyLookup).forEach((companyKey) => {
    currentCompanySymbols.push(companyLookup[companyKey]);
  });

  let allCompanyMetrics = awaitmetricsRetrieve.retrieveCompanies());

  allCompanyMetrics.forEach((companyMetric) => {
    currentMetricsCompanies.push(companyMetric['symbol']);
  });

  currentCompanySymbols.forEach((companySymbol) => {
    if (currentMetricsCompanies.indexOf(companySymbol) === -1) {
      companiesToRemove.push(companySymbol);
    }
  });

  utils.writeJSONfile(companiesToRemove, '../data/companies-to-remove.json');
});

let verifyCompaniesList = asyncfunction() {
  let queryDetails = {
    tableName: 'companyQuotes',
    keyConditionExpression: 'symbol = :symbol and ' +
      'quoteDate >= :quoteDate',
    expressionAttributeValues: {
      ':quoteDate': '2017-01-25',
    },
    limit: 1,
  };

  let verifiedRemovalList = [];

  jsonCompanies.forEach((symbol) => {
    queryDetails['expressionAttributeValues'][':symbol'] = symbol;
    let indexQuotes = awaitdynamodb.queryTable(queryDetails));

    // if no records, then verify that this should be removed
    if (!indexQuotes.length) {
      verifiedRemovalList.push(symbol);
    }
  });

  utils.writeJSONfile(verifiedRemovalList, '../data/verified-companies-to-remove.json');
});


let removeCompanies = asyncfunction() {
  let deleteDetails = {
    tableName: 'companies',
  };

  jsonCompanies.forEach((symbol) => {
    deleteDetails.key = {
      'symbol': symbol,
    };

    awaitdynamodb.deleteRecord(deleteDetails));
  });
});

let removeMetrics = asyncfunction() {
  let deleteDetails = {
    tableName: 'companyMetrics',
  };

  let queryDetails = {
    tableName: 'companyMetrics',
    keyConditionExpression: 'symbol = :symbol and ' +
      'metricsDate >= :metricsDate',
    expressionAttributeValues: {
      ':metricsDate': '2017-01-25',
    },
    projectionExpression: 'symbol, metricsDate',
  };

  jsonCompanies.forEach((symbol) => {
    queryDetails['expressionAttributeValues'][':symbol'] = symbol;

    let queryResults = awaitdynamodb.queryTable(queryDetails));

    queryResults.forEach((result) => {
      deleteDetails.key = {
        'symbol': symbol,
        'metricsDate': result['metricsDate'],
      };

      awaitdynamodb.deleteRecord(deleteDetails));
    });
  });
}); */

/**  Write a conpany quote record to dynamodb.
*    Converts lastTradeDate -> quoteDate, copies lastTradePriceOnly ->
*     adjustedPrice, and checks and removes any invalid values.
* @param {Object} quoteData - the company quote to write
*/
let writeCompanyQuoteData = asyncfunction(quoteData) {
  // If unexpected r3cords come back with no trade data, skop them
  if (!quoteData['lastTradeDate']) {
    return;
  }

  let upsertDetails = {
    tableName: 'companyQuotes',
  };

  quoteData.quoteDate = utils.returnDateAsString(quoteData['lastTradeDate']);
  quoteData.yearMonth = quoteData['quoteDate'].substring(0, 7).replace('-', '');
  if (!quoteData.adjustedPrice) {
    quoteData.adjustedPrice = quoteData['lastTradePriceOnly'];
  }

  delete quoteData['lastTradeDate'];

  // Check through for values with null and remove from object
  Object.keys(quoteData).forEach((field) => {
    if (quoteData[field] === null || quoteData[field] === '') {
      delete quoteData[field];
    }
  });

  upsertDetails.key = {
    symbol: quoteData.symbol,
    quoteDate: quoteData.quoteDate,
  };

  let updateExpression;
  let expressionAttributeValues = {};
  let expressionAttributeNames = {};
  let workingRecord = {};
  let fieldsPresent = [];

  Object.assign(workingRecord, quoteData);

  // Remove the attributes which should not be update in comanyQuotes
  delete workingRecord['symbol'];
  delete workingRecord['quoteDate'];
  delete workingRecord['created'];

  // Separately set-up creeated attributed values and names
  expressionAttributeValues[':created'] = moment().tz('Australia/Sydney').format();
  expressionAttributeNames['#created'] = 'created';

  // Get a list of fields and values to copy
  Object.keys(workingRecord).forEach((field) => {
    expressionAttributeValues[(':' + field)] = workingRecord[field];
    expressionAttributeNames[('#' + field)] = field;
    fieldsPresent.push('#' + field + '=:' + field);
  });

  // Enure that at least one field is present to update
  if (fieldsPresent.length) {
    updateExpression = 'set ' + fieldsPresent.join(',');
    // Add in special handing\ling for the created field
    updateExpression = updateExpression +
      ', #created = if_not_exists (#created, :created)';

    upsertDetails.updateExpression = updateExpression;
    upsertDetails.expressionAttributeValues = expressionAttributeValues;
    upsertDetails.expressionAttributeNames = expressionAttributeNames;

    return awaitdynamodb.updateRecord(upsertDetails));
  }
});

let checkExclusionTime = function(startExlusionTime, endEclusionTime) {
  if (!moment(startExlusionTime, 'HH:mm').isValid()) {
    return false;
  }

  if (!moment(endEclusionTime, 'HH:mm').isValid()) {
    return false;
  }
  let parsedNow = moment().tz('Australia/Sydney').format('HH:mm');

  if (parsedNow >= startExlusionTime && parsedNow <= endEclusionTime) {
    return true;
  } else {
    return false;
  }
};


getCompanyHistory();
