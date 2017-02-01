const yahooFinance = require('yahoo-finance');
const utils = require('./utils');

const fields = {
  // b: 'bid',
  // a: 'ask',
  p: 'previous-close',
  o: 'open',
  y: 'dividend-yield',
  d: 'dividend-per-share',
  // r1: 'dividend-pay-date',
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
  // t8: '1-yr-target-price',
  // m5: 'change-from-200-day-moving-average',
  // m6: 'percent-change-from-200-day-moving-average',
  // m7: 'change-from-50-day-moving-average',
  // m8: 'percent-change-from-50-day-moving-average',
  // m3: '50-day-moving-average',
  // m4: '200-day-moving-average',
  // w1: 'day-value-change',
  // p1: 'price-paid',
  // m: 'day-range',
  // g1: 'holdings-gain-percent',
  // g3: 'annualized-gain',
  // g4: 'holdings-gain',
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
  // e8: 'EPS-estimate-next-year',
  // e9: 'EPS-estimate-next-quarter',
  b4: 'book-value',
  j4: 'EBITDA',
  p5: 'price-per-sales',
  p6: 'price-per-book',
  r: 'PE-ratio',
  r5: 'PEG-ratio',
  r6: 'price-per-EPS-estimate-current-year',
  r7: 'price-per-EPS-estimate-next-year',
  s7: 'short-ratio',
  // t7: 'ticker-trend',
  // t6: 'trade-links',
  // l2: 'high-limit',
  // l3: 'low-limit',
  // v1: 'holdings-value',
  s6: 'revenue',
};

/* const realTimeFields = {
  c6: 'change-realtime',
  k2: 'change-percent-realtime',
  c8: 'after-hours-change-realtime',
  k1: 'last-trade-realtime-with-time',
  w4: 'day-value-change-realtime',
  m2: 'day-range-realtime',
  g5: 'holdings-gain-percent-realtime',
  g6: 'holdings-gain-realtime',
  j3: 'market-cap-realtime',
  r2: 'PE-ratio-realtime',
  i5: 'order-book-realtime',
  v7: 'holdings-value-realtime',
}; */

const indiceFields = {
  d1: 'last-trade-date',
  p: 'previous-close',
  o: 'open',
  c1: 'change',
  c: 'change-and-percent-change',
  p2: 'change-in-percent',
  g: 'day-low',
  h: 'day-high',
  // w1: 'day-value-change',
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
const lastResultDate = utils.getLastRetrievalDate();
const indiceFieldsToRetrieve = utils.createFieldArray(indiceFields);
const companyFieldsToRetrieve = utils.createFieldArray(fields);
let symbolGroups = [];
let shareRetrievals = [];
let csvFields = [];
let csvData = [];
let maxResultDate = '';

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


let retrieveSnapshot = function(symbol, fields) {
  return new Promise(function(resolve, reject) {
    let snapshotOptions = {
      fields: fields,
    };

    // Check if one or many symbols
    if (Array.isArray(symbol)) {
      snapshotOptions.symbols = symbol;
    } else {
      snapshotOptions.symbol = symbol;
    }
    yahooFinance.snapshot(snapshotOptions).then(function(result) {
      resolve(result);
    }).catch(function(err) {
      reject(err);
    });
  });
};

let processResults = function(results) {
  if (results) {
    // Check if multi-dimensional
    if (Array.isArray(results)) {
      // Multiple  symbols returned
      results.forEach(function(indResult) {
        processResult(indResult);
      });
    } else {
      // Single symbol returned
      processResult(results);
    }
  }
};

let processResult = function(result) {
  // Retrieve last trade date to check whether to output this value
  result.lastTradeDate = utils.returnDateAsString(result.lastTradeDate);
  if (result.lastTradeDate > lastResultDate) {
    if (result.lastTradeDate > maxResultDate) {
      maxResultDate = result.lastTradeDate;
    }

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
  }
};

setupSymbols();


retrieveSnapshot(indices, indiceFieldsToRetrieve)
  .then(function(results) {
    return processResults(results);
  })
  .then(function() {
    if (csvData.length > 0) {
      utils.writeToCsv(csvData, csvFields, 'indices', maxResultDate);
    } else {
      console.log('No new index data to save');
    }

    return true;
  }).then(function() {
  // Reset fields for companies
  csvFields = [];
  csvData = [];

  // Split companies into groups of 10 so each request contains 10
  for (companyCounter = 0; companyCounter < companies.length;
    companyCounter += 10) {
    symbolGroups.push(companies.slice(companyCounter, companyCounter + 10));
  }

  symbolGroups.forEach(function(symbolGroup) {
    shareRetrievals.push(retrieveSnapshot(symbolGroup,
      companyFieldsToRetrieve));
  });

  // When all returns are back, process the results
  Promise.all(shareRetrievals).then(function(results) {
    results.forEach(function(result) {
      processResults(result);
    });

    if (csvData.length > 0) {
      utils.writeToCsv(csvData, csvFields, 'companies', maxResultDate);

      // Re-write JSON file with last retrieval date
      lastRetrievalDate = {
        'retrieval-date': maxResultDate,
      };
      utils.writeJSONfile(lastRetrievalDate, 'last-retrieval.json');
    } else {
      console.log('No new company data to save');
    }
  }).catch(function(err) {
    console.log(err);
  });
})
  .catch(function(err) {
    console.log(err);
  });
