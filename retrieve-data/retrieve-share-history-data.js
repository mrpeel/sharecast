const yahooFinance = require('yahoo-finance');
const utils = require('./utils');

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
let csvFields = [];
let csvData = [];

let setupSymbols = function() {
  let indexValues = utils.getIndices();
  let companyValues = utils.getCompanies();

  indexValues.forEach(function(indexValue) {
    indexLookup[indexValue['yahoo-symbol']] = indexValue['symbol'];
    symbolLookup[indexValue['yahoo-symbol']] = indexValue['symbol'];
  });

  indices = utils.createFieldArray(indexLookup);

  companyValues.forEach(function(companyValue) {
    companyLookup[companyValue['yahoo-symbol']] = companyValue['symbol'];
    symbolLookup[companyValue['yahoo-symbol']] = companyValue['symbol'];
  });

  companies = utils.createFieldArray(companyLookup);
};

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


let processHistoryResults = function(results) {
  if (results) {
    Object.keys(results).forEach(function(symbolResults) {
      // Multiple  symbols returned
      results[symbolResults].forEach(function(indResult) {
        processHistoryResult(indResult);
      });
    });
  }
};

let processHistoryResult = function(result) {
  result.date = utils.returnDateAsString(result.date);
  // Convert yahoo symbol to generic symbol
  result.symbol = symbolLookup[result.symbol];

  Object.keys(result).forEach(function(field) {
    // Check the field is in the csv list
    if (csvFields.indexOf(field) === -1) {
      csvFields.push(field);
    }

    // Reset number here required
    result[field] = utils.checkForNumber(result[field]);
  });
  // Add result to csv data
  csvData.push(result);
};

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
}

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

/*
// Split companies into groups of 10 so each request contains 10
for (companyCounter = 0; companyCounter < companies.length;
  companyCounter += 10) {
  // symbolGroups.push(companies.slice(companyCounter, companyCounter + 10));
  let companySymbols = companies.slice(companyCounter, companyCounter + 10);
  let internalCounter = companyCounter;

  if (!utils.doesDataFileExist('companies-history-' + internalCounter + '-'
      + dateString + '.csv')) {
    setTimeout(function() {
      console.log('Processing companyCounter: ' + internalCounter);
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
    }, internalCounter * 20);
  } else {
    console.log('Skipping companyCounter: ' + internalCounter);
  }
}

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
