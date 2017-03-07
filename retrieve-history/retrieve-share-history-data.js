const yahooFinance = require('yahoo-finance');
const utils = require('./utils');
const dynamodb = require('./dynamodb');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const shareRetrieve = require('./dynamo-retrieve-share-data');
const metricsRetrieve = require('./dynamo-retrieve-google-company-data');
const fiRetrieve = require('./dynamo-retrieve-financial-indicator-data');
const stats = require('./stats');
const jsonCompanies = require('../data/verified-companies-to-remove.json');

const fields = {
  // b: 'bid',
  // a: 'ask',
  p: 'previous-close',
  o: 'open',
  y: 'dividend-yield',
  d: 'dividend-per-share',
  r1: 'dividend-pay-date',
  q: 'ex-dividend-date',
  c1: 'change',
  // c: 'change-and-percent-change',
  p2: 'change-in-percent',
  d1: 'last-trade-date',
  // d2: 'trade-date',
  // t1: 'last-trade-time',
  // c3: 'commission',
  g: 'day-low',
  h: 'day-high',
  // l: 'last-trade-with-time',
  l1: 'last-trade-price-only',
  t8: '1-yr-target-price',
  // m5: 'change-from-200-day-moving-average',
  // m6: 'percent-change-from-200-day-moving-average',
  // m7: 'change-from-50-day-moving-average',
  // m8: 'percent-change-from-50-day-moving-average',
  // m3: '50-day-moving-average',
  // m4: '200-day-moving-average',
  w1: 'day-value-change',
  p1: 'price-paid',
  // m: 'day-range',
  g1: 'holdings-gain-percent',
  g3: 'annualized-gain',
  g4: 'holdings-gain',
  k: '52-week-high',
  j: '52-week-low',
  // j5: 'change-from-52-week-low',
  // k4: 'change-from-52-week-high',
  // j6: 'percent-change-from-52-week-low',
  // k5: 'percent-change-from-52-week-high',
  // w: '52-week-range',
  j1: 'market-capitalization',
  f6: 'float-shares',
  // n: 'name',
  // n4: 'notes',
  s1: 'shares-owned',
  x: 'stock-exchange',
  j2: 'shares-outstanding',
  v: 'volume',
  // a5: 'ask-size',
  // b6: 'bid-size',
  // k3: 'last-trade-size',
  // a2: 'average-daily-volume',
  e: 'earnings-per-share',
  e7: 'EPS-estimate-current-year',
  e8: 'EPS-estimate-next-year',
  e9: 'EPS-estimate-next-quarter',
  b4: 'book-value',
  j4: 'EBITDA',
  p5: 'price-per-sales',
  p6: 'price-per-book',
  r: 'PE-ratio',
  r5: 'PEG-ratio',
  r6: 'price-per-EPS-estimate-current-year',
  r7: 'price-per-EPS-estimate-next-year',
  s7: 'short-ratio',
  t7: 'ticker-trend',
  // t6: 'trade-links',
  // l2: 'high-limit',
  // l3: 'low-limit',
  v1: 'holdings-value',
  s6: 'revenue',
};

const indiceFields = {
  d1: 'last-trade-date',
  p: 'previous-close',
  o: 'open',
  c1: 'change',
  c: 'change-and-percent-change',
  p2: 'change-in-percent',
  g: 'day-low',
  h: 'day-high',
  w1: 'day-value-change',
  m: 'day-range',
  k: '52-week-high',
  j: '52-week-low',
  j5: 'change-from-52-week-low',
  k4: 'change-from-52-week-high',
  j6: 'percent-change-from-52-week-low',
  k5: 'percent-change-from-52-week-high',
  n: 'name',
  x: 'stock-exchange',
  j2: 'shares-outstanding',
};

let symbolLookup = {};
let indexLookup = {};
let companyLookup = {};
let indices = [];
let companies = [];

/* Retrieve index values */
const indiceFieldsToRetrieve = utils.createFieldArray(indiceFields);
const companyFieldsToRetrieve = utils.createFieldArray(fields);
// let symbolGroups = [];
// let shareRetrievals = [];
let dataFields = [];
let data = [];

let retrieveHistory = function(symbol, fields, startDate, endDate, interval) {
  return new Promise(function(resolve, reject) {
    let historyOptions = {
      fields: fields,
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

/**
 * Returns a dividend value for a company date combination (if exists)
 * specific date
 * @param {String} companySymbol the companySymbol to look up
 * @param {String} valueDate the date to check
 * @return {Number}  the dividend value / null if not found
 */
let returnDividendValueForDate = asyncify(function(companySymbol, valueDate) {
  if (!valueDate || !utils.isDate(valueDate)) {
    throw new Error('valueDate supplied is invalid: ' + valueDate);
  }

  if (!companySymbol) {
    throw new Error('companySymbol not supplied');
  }

  let connection;
  try {
    // Open DB connection
    connection = awaitify(dbConn.connectToDb(host, username, password, db));

    let result = awaitify(dbConn.selectQuery(connection,
      'SELECT `value` ' +
      'FROM `sharecast`.`dividend_history` ' +
      'WHERE `company_symbol` = \'' + companySymbol + '\' ' +
      'AND `dividend_date` <= \'' + valueDate + '\' ' +
      'ORDER BY `dividend_date` desc ' +
      'LIMIT 1;'
    ));
    if (result.length > 0) {
      return result[0]['value'];
    } else {
      return null;
    }
  } catch (err) {
    console.log(err);
  } finally {
    if (connection) {
      dbConn.closeConnection(connection);
    }
  }
});


let getDividendHistory = asyncify(function() {
  let symbolResult = awaitify(shareRetrieve.setupSymbols());

  symbolLookup = symbolResult.symbolLookup;
  indexLookup = symbolResult.indexLookup;
  indexSymbols = symbolResult.indexSymbols;
  companyLookup = symbolResult.companyLookup;
  indices = symbolResult.indices;
  companies = symbolResult.companies;

  // Split companies into groups of 10 so each request contains 10
  for (companyCounter = 0; companyCounter < companies.length;
    companyCounter += 10) {
    // symbolGroups.push(companies.slice(companyCounter, companyCounter + 10));
    let companySymbols = companies.slice(companyCounter, companyCounter + 10);
    let internalCounter = companyCounter;

    let insertDetails = {
      tableName: 'companyMetrics',
      primaryKey: ['symbol', 'metricsDate'],
    };


    console.log('Processing companyCounter: ' + internalCounter);
    /* awaitify(retrieveHistory(companySymbols, companyFieldsToRetrieve,
      '2006-01-01', '2017-01-29', 'd') */
    try {
      let results = awaitify(retrieveDividendHistory(companySymbols,
        '2006-01-01', '2017-01-29'));
      // Reset fields for companies
      data = [];

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

/* retrieveHistory(indices, indiceFieldsToRetrieve, '2012-01-01',
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
    retrieveHistory(companySymbols, companyFieldsToRetrieve, '2012-01-01',
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

  symbolLookup = symbolResult.symbolLookup;
  indexLookup = symbolResult.indexLookup;
  indexSymbols = symbolResult.indexSymbols;
  companyLookup = symbolResult.companyLookup;
  indices = symbolResult.indices;
  companies = symbolResult.companies;

  let results = awaitify(retrieveHistory(indices, indiceFieldsToRetrieve,
    '2006-07-01', '2011-12-31', 'd'));

  processIndexHistoryResults(results);
});


let getCompanyHistory = asyncify(function() {
  let symbolResult = awaitify(shareRetrieve.setupSymbols());

  symbolLookup = symbolResult.symbolLookup;
  indexLookup = symbolResult.indexLookup;
  indexSymbols = symbolResult.indexSymbols;
  companyLookup = symbolResult.companyLookup;
  indices = symbolResult.indices;
  companies = symbolResult.companies;

  // Work through companies one by one and retrieve values
  companies.forEach((companySymbol) => {
    let results = awaitify(retrieveHistory(companySymbol,
      companyFieldsToRetrieve, '2006-07-01', '2017-01-29', 'd'));

    processCompanyHistoryResults(results);
  });
});

let processCompanyHistoryResults = asyncify(function(results) {
  if (results) {
    Object.keys(results).forEach(function(symbolResults) {
      // Multiple  symbols returned
      results[symbolResults].forEach(function(indResult) {
        awaitify(processCompanyHistoryResult(indResult));
      });
    });
  }
});

let processCompanyHistoryResult = asyncify(function(result) {
  let ignoreVals = ['created', 'yearMonth', 'valueDate', 'metricsDate',
    'quoteDate'];
  result.lastTradeDate = utils.returnDateAsString(result.date);
  // Convert yahoo symbol to generic symbol
  result.symbol = symbolLookup[result.symbol];

  // Remove oriingal date field
  delete result['date'];

  Object.keys(result).forEach(function(field) {
    // Reset number here required
    result[field] = utils.checkForNumber(result[field]);
  });

  // Get index values for date
  let indexVals = awaitify(
    shareRetrieve.returnIndexDataForDate(result.lastTradeDate));

  Object.keys(indexVals).forEach((indexKey) => {
    // Check if value should be ignored.  If not add to quoteVal
    if (ignoreVals.indexOf(indexKey) === -1) {
      result[indexKey] = indexVals[indexKey];
    }
  });

  // Get financial indicator values for date
  let fiVals = awaitify(
    fiRetrieve.returnIndicatorValuesForDate(result.lastTradeDate));

  Object.keys(fiVals).forEach((fiKey) => {
    // Check if value should be ignored.  If not add to quoteVal
    if (ignoreVals.indexOf(fiKey) === -1) {
      result[fiKey] = fiVals[fiKey];
    }
  });

  // Get metric values for date
  let metricsVals = awaitify(
    metricsRetrieve.returnCompanyMetricValuesForDate(result.lastTradeDate));

  Object.keys(metricsVals).forEach((metricsKey) => {
    // Check if value should be ignored.  If not add to quoteVal
    if (ignoreVals.indexOf(metricsKey) === -1) {
      result[metricsKey] = metricsVals[metricsKey];
    }
  });

  // Insert result in dynamodb
  awaitify(shareRetrieve.writeCompanyQuoteData(result));
});

/* companyQuote data
History
Date -> quoteDate, lastTradeDate
Open	-> open
High -> daysHigh
Low -> daysLow
Close	-> lastTradePriceOnly
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

  symbolLookup = symbolResult.symbolLookup;
  indexLookup = symbolResult.indexLookup;
  indexSymbols = symbolResult.indexSymbols;
  companyLookup = symbolResult.companyLookup;
  indices = symbolResult.indices;
  companies = symbolResult.companies;

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

removeMetrics();

// getIndexHistory();

// getCompanyHistory();

// addDailyChangeToIndexHistory();
