const yahooFinance = require('yahoo-finance');
const retrieval = require('./retrieval.json');
const utils = require('./utils.json');
const json2csv = require('json2csv');
const fs = require('fs');

// const googleFinance = require('google-finance');
// const exchange = 'AX';
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
  m: 'day-range',
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
  n: 'name',
  n4: 'notes',
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

const realTimeFields = {
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
};

const indiceFields = {
  p: 'previous-close',
  o: 'open',
  c1: 'change',
  c: 'change-and-percent-change',
  p2: 'change-in-percent',
  g: 'day-low',
  h: 'day-high',
  l: 'last-trade-with-time',
  l1: 'last-trade-price-only',
  w1: 'day-value-change',
  m: 'day-range',
  k: '52-week-high',
  j: '52-week-low',
  j5: 'change-from-52-week-low',
  k4: 'change-from-52-week-high',
  j6: 'percent-change-from-52-week-low',
  k5: 'percent-change-from-52-week-high',
  w: '52-week-range',
  n: 'name',
  x: 'stock-exchange',
  j2: 'shares-outstanding',
  v: 'volume',
  a2: 'average-daily-volume',
};

const shareIndices = {
  '^AXJO': 'ASX',
  '^AORD': 'AllOrdinaries',
};

let createFieldArray = function(fieldObject) {
  return Object.keys(fieldObject);
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

/* period: 'd'  // 'd' (daily), 'w' (weekly), 'm' (monthly),
'v' (dividends only)
*/
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


let outputResults = function(results) {
  if (results) {
    // Check if multi-dimensional
    if (Array.isArray(results)) {
      console.log('Multiple results');
      results.forEach(function(indResult) {
        Object.keys(indResult).forEach(function(result) {
          console.log(result + ': ' + utils.checkForNumber(indResult[result]));
        });
      });
    } else {
      Object.keys(results).forEach(function(result) {
        console.log(result + ': ' + utils.checkForNumber(results[result]));
      });
    }
  }
};


let indicesToRetrieve = createFieldArray(shareIndices);
let indiceFieldsToRetrieve = createFieldArray(indiceFields);

retrieveSnapshot(indicesToRetrieve, indiceFieldsToRetrieve)
  .then(function(results) {
    console.log('----------  Indices  ----------');
    outputResults(results);
    return true;
  })
  .then(function() {
    let fieldsToRetrieve = createFieldArray(fields);
    return retrieveSnapshot('DUE.AX', fieldsToRetrieve);
  })
  .then(function(results) {
    console.log('----------  Normal shares  ----------');
    outputResults(results);
  });

let fieldsToRetrieve = createFieldArray(fields);
retrieveHistory('DUE.AX', fieldsToRetrieve, '2016-01-01', '2016-01-10', 'd')
  .then(function(results) {
    console.log('----------  History results  ----------');
    outputResults(results);
  });
