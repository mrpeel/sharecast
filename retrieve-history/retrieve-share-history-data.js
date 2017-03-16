const yahooFinance = require('yahoo-finance');
const utils = require('../retrieve-data/utils');
const dynamodb = require('../retrieve-data/dynamodb');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const shareRetrieve = require('../retrieve-data/dynamo-retrieve-share-data');
const metricsRetrieve = require('../retrieve-data/dynamo-retrieve-google-company-data');
const fiRetrieve = require('../retrieve-data/dynamo-retrieve-financial-indicator-data');
const stats = require('../retrieve-data/stats');
const processRollups = require('../retrieve-data/process-rollups');
const histSymbols = require('./symbols.json');
const jsonCompanies = require('../data/verified-companies-to-remove.json');

let indexValsLookup = {};
let fiValsLookup = {};

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

let retrieveDividendHistory = function(symbol, startDate, endDate) {
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
};


let processIndexHistoryResults = asyncify(function(results) {
  if (results) {
    Object.keys(results).forEach(function(symbolResults) {
      // Multiple  symbols returned
      results[symbolResults].forEach(function(indResult) {
        awaitify(processIndexHistoryResult(indResult));
      });
    });
  }
});

let processIndexHistoryResult = asyncify(function(result) {
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
  awaitify(shareRetrieve.writeIndexQuote(result));
});

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


let getDividendHistory = asyncify(function() {
  let symbolResult = awaitify(shareRetrieve.setupSymbols());

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
    /* awaitify(retrieveHistory(companySymbols,
      '2006-01-01', '2017-01-29', 'd') */
    try {
      let results = awaitify(retrieveDividendHistory(companySymbols,
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

          awaitify(dynamodb.insertRecord(insertDetails));
        });
      } else {
        console.log('No history data to save');
      }
    } catch (err) {
      console.log(err);
    }
  }
});

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

let getIndexHistory = asyncify(function() {
  let symbolResult = awaitify(shareRetrieve.setupSymbols());

  let symbolLookup = symbolResult.symbolLookup;
  let indexLookup = symbolResult.indexLookup;
  let indexSymbols = symbolResult.indexSymbols;
  let companyLookup = symbolResult.companyLookup;
  let indices = symbolResult.indices;
  let companies = symbolResult.companies;

  let results = awaitify(retrieveHistory(indices,
    '2006-07-01', '2011-12-31', 'd'));

  processIndexHistoryResults(results);
});


let getCompanyHistory = asyncify(function() {
  let currentCompany;
  try {
    dynamodb.setLocalAccessConfig();
    // override console within this block
    console.log = function() {};
    let symbolResult = setupHistorySymbols();

    let symbolLookup = symbolResult.symbolLookup;
    let companies = symbolResult.companies;

    // Work through companies one by one and retrieve values
    companies.forEach((companySymbol) => {
      // Skip previous companies
      if (companySymbol >= 'CJT.AX') {
        let results = awaitify(retrieveHistory(companySymbol,
          '2006-07-01', '2007-06-30', 'd'));

        awaitify(processCompanyHistoryResults(results, symbolLookup));
      }
    });
  } catch (err) {
    console.log('Failed while processing: ', currentCompany);
    console.log(err);
  }
});

let processCompanyHistoryResults = asyncify(function(results, symbolLookup) {
  if (results) {
    Object.keys(results).forEach(function(symbolResults) {
      /* Single symbol returned - check whether this is a normal record,
          i.e. has Close val */
      if (results[symbolResults].close) {
        awaitify(processCompanyHistoryResult(results[symbolResults],
          symbolLookup));
      }
    });
  }
});

let processCompanyHistoryResult = asyncify(function(result, symbolLookup) {
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
    result.changeInPercent = (result.close - result.open) / result.open;
  }

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

  // Calculate 1, 2, 4, 8, 12, 26, 52 week cut-off dateString

  // params: todayDate

  // Get dividend values for last 52 weeks

  // params: 52WeekCutOffDate


  // Retrieve values for 1, 2, 4, 8, 12, 26, 52 week adjusted prices

  // params: allAdjustedPrices

  // Calculate average for 1, 2, 4, 8,12, 26, 52 weeks

  // params: arrayOfAdjustedPrices

  // Calculate std deviation for 1, 2, 4, 8, 12, 26, 52 weeks

  // params: arrayOfAdjustedPrices for each time period

  // Calculate 4, 12 week Bollinger bands

  // Calculate 52 week high, low, change and percentChange

  // params: 52WeekAdjustedPrices, currentPrice

  // params: numberWeeks, currentPrice, std deviation

  /* Calculate and update total return and risk adjusted return
    for 1, 2, 4, 8, 12, 26, 52 weeks */

  // params: numberWeeks, targetDate, currentPrice, dividends, std deviation


  // Get index values for date
  if (!indexValsLookup[result.lastTradeDate]) {
    let indexValRetrieve = awaitify(
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

  // Get financial indicator values for date
  if (!fiValsLookup[result.lastTradeDate]) {
    let fiValsRetrieve = awaitify(
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


  // Get metric values for date
  let metricsVals = awaitify(
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

  // Insert result in dynamodb
  awaitify(shareRetrieve.writeCompanyQuoteData(result));

// Add current quote to company adjusted price array
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

let fixIndexHistory = asyncify(function() {
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


  let indexQuotes = awaitify(dynamodb.queryTable(queryDetails));

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
      awaitify(dynamodb.updateRecord(updateDetails));
    } catch (err) {
      console.log(err);
    }
  });
});

let addYearResultsToIndexHistory = asyncify(function() {
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


    let indexQuotes = awaitify(dynamodb.queryTable(queryDetails));

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

      let quotesYear = awaitify(dynamodb.queryTable(query2Details));

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
          awaitify(dynamodb.updateRecord(updateDetails));
        } catch (err) {
          console.log(err);
        }
      }
    });
  });
});


let addDailyChangeToIndexHistory = asyncify(function() {
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


    let indexQuotes = awaitify(dynamodb.queryTable(queryDetails));

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

      let yesterdayQuote = awaitify(dynamodb.queryTable(query2Details));

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
          awaitify(dynamodb.updateRecord(updateDetails));
        } catch (err) {
          console.log(err);
        }
      }
    });
  });
});

let fixCompaniesList = asyncify(function() {
  let currentCompanySymbols = [];
  let companiesToRemove = [];
  let currentMetricsCompanies = [];
  let symbolResult = awaitify(shareRetrieve.setupSymbols());

  let symbolLookup = symbolResult.symbolLookup;
  let indexLookup = symbolResult.indexLookup;
  let indexSymbols = symbolResult.indexSymbols;
  let companyLookup = symbolResult.companyLookup;
  let indices = symbolResult.indices;
  let companies = symbolResult.companies;

  Object.keys(companyLookup).forEach((companyKey) => {
    currentCompanySymbols.push(companyLookup[companyKey]);
  });

  let allCompanyMetrics = awaitify(metricsRetrieve.retrieveCompanies());

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

let verifyCompaniesList = asyncify(function() {
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
    let indexQuotes = awaitify(dynamodb.queryTable(queryDetails));

    // if no records, then verify that this should be removed
    if (!indexQuotes.length) {
      verifiedRemovalList.push(symbol);
    }
  });

  utils.writeJSONfile(verifiedRemovalList, '../data/verified-companies-to-remove.json');
});


let removeCompanies = asyncify(function() {
  let deleteDetails = {
    tableName: 'companies',
  };

  jsonCompanies.forEach((symbol) => {
    deleteDetails.key = {
      'symbol': symbol,
    };

    awaitify(dynamodb.deleteRecord(deleteDetails));
  });
});

let removeMetrics = asyncify(function() {
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

    let queryResults = awaitify(dynamodb.queryTable(queryDetails));

    queryResults.forEach((result) => {
      deleteDetails.key = {
        'symbol': symbol,
        'metricsDate': result['metricsDate'],
      };

      awaitify(dynamodb.deleteRecord(deleteDetails));
    });
  });
});

getCompanyHistory();

// getIndexHistory();

// getCompanyHistory();

// addDailyChangeToIndexHistory();
