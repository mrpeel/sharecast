const yahooFinance = require('yahoo-finance');
const utils = require('./utils');
const dynamodb = require('./dynamodb');
const symbols = require('./dynamo-symbols');
const finIndicators = require('./dynamo-retrieve-financial-indicator-data');
const metrics = require('./dynamo-retrieve-google-company-data');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const program = require('commander');

const fields = {
  p: 'previous-close',
  y: 'dividend-yield',
  d: 'dividend-per-share',
  q: 'ex-dividend-date',
  r1: 'dividend-pay-date',
  c1: 'change',
  p2: 'change-in-percent',
  d1: 'last-trade-date',
  g: 'day-low',
  h: 'day-high',
  l1: 'last-trade-price-only',
  k: '52-week-high',
  j: '52-week-low',
  j1: 'market-capitalization',
  f6: 'float-shares',
  s1: 'shares-owned',
  x: 'stock-exchange',
  j2: 'shares-outstanding',
  v: 'volume',
  e: 'earnings-per-share',
  e7: 'EPS-estimate-current-year',
  b4: 'book-value',
  j4: 'EBITDA',
  p5: 'price-per-sales',
  p6: 'price-per-book',
  r: 'PE-ratio',
  r5: 'PEG-ratio',
  r6: 'price-per-EPS-estimate-current-year',
  r7: 'price-per-EPS-estimate-next-year',
  s7: 'short-ratio',
  s6: 'revenue',
};

const indiceFields = {
  d1: 'last-trade-date',
  p: 'previous-close',
  c1: 'change',
  p2: 'change-in-percent',
  g: 'day-low',
  h: 'day-high',
  k: '52-week-high',
  j: '52-week-low',
  j5: 'change-from-52-week-low',
  k4: 'change-from-52-week-high',
  j6: 'percent-change-from-52-week-low',
  k5: 'percent-change-from-52-week-high',
  n: 'name',
};

/* let metricsFields = ['EPS', 'PriceToBook', 'MarketCap', 'PE',
  'DividendRecentQuarter', 'DividendNextQuarter', 'DPSRecentYear',
  'IAD', 'DividendPerShare', 'DividendYield', 'Dividend',
  'BookValuePerShareYear', 'CashPerShareYear', 'CurrentRatioYear',
  'LTDebtToAssetsYear', 'LTDebtToAssetsQuarter',
  'TotalDebtToAssetsYear', 'TotalDebtToAssetsQuarter',
  'LTDebtToEquityYear', 'LTDebtToEquityQuarter',
  'TotalDebtToEquityYear', 'TotalDebtToEquityQuarter', 'AINTCOV',
  'ReturnOnInvestmentTTM', 'ReturnOnInvestment5Years',
  'ReturnOnInvestmentYear', 'ReturnOnAssetsTTM', 'ReturnOnAssets5Years',
  'ReturnOnAssetsYear', 'ReturnOnEquityTTM', 'ReturnOnEquity5Years',
  'ReturnOnEquityYear', 'Beta', 'Float', 'GrossMargin', 'EBITDMargin',
  'OperatingMargin', 'NetProfitMarginPercent',
  'NetIncomeGrowthRate5Years', 'RevenueGrowthRate5Years',
  'RevenueGrowthRate10Years', 'EPSGrowthRate5Years',
  'EPSGrowthRate10Years']; */

/* Retrieve index values */
const indiceFieldsToRetrieve = utils.createFieldArray(indiceFields);
const companyFieldsToRetrieve = utils.createFieldArray(fields);

/**  Retrieve company and index symbols and sets them up for retrieval
* @return {Object} coninating values in the format:
*  {
*    indexLookup: {Object}, (object to match index symbol with lookup
*                          symbol, e.g. ^ALLORD: ALLORD)
*    indices: [], (array of symbols to use during retrieval)
*    companyLookup: {Object},  (object to match company symbol with lookup
*                          symbol ^AAD: AAD)
*    companies: [], (array of symbols to use during retrieval)
*    symbolLookup: {Object}, (object to match lookup with real index / company
*                          symbol)
*    indexSymbols: [], (array of the index symbols used)
*  }
*/
let setupSymbols = asyncify(function(indicesOnly) {
  try {
    console.log('----- Start setup symbols -----');
    let indexValues = awaitify(symbols.getIndices());
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

    if (!indicesOnly) {
      let companyValues = awaitify(symbols.getCompanies());

      companyValues.forEach((companyValue) => {
        wCompanyLookup[companyValue['yahoo-symbol']] = companyValue['symbol'];
        wSymbolLookup[companyValue['yahoo-symbol']] = companyValue['symbol'];
      });

      wCompanies = utils.createFieldArray(wCompanyLookup);

      returnVal.companyLookup = wCompanyLookup;
      returnVal.companies = wCompanies;
    }

    return returnVal;
  } catch (err) {
    console.log(err);
  }
});

/**  Use the max date for an array of index symbols
*     to determine the last date which data was collected
* @param {Array} indexSymbols - one or more index symbols to look up
* @return {String} with date formatted as 'YYYY-MM-DD'
*/
/* let getLastRetrievalDate = function(indexSymbols) {
  console.log('----- Start get last retrieval date -----');
  return new Promise(function(resolve, reject) {
    try {
      let queryDetails = {
        tableName: 'indexQuotes',
        keyConditionExpression: 'symbol = :symbol',
        reverseOrder: true,
        limit: 1,
        projectionExpression: 'quoteDate',
      };

      let retrievalDate = '';


      indexSymbols.forEach((indexSymbol) => {
        queryDetails.expressionAttributeValues = {
          ':symbol': indexSymbol,
        };
        let result = awaitify(dynamodb.queryTable(queryDetails));

        // Check if this result is later than what we already have
        if (result.length > 0 && result[0]['quoteDate'] > retrievalDate) {
          retrievalDate = result[0]['quoteDate'];
        }
      });

      resolve(retrievalDate);
    } catch (err) {
      console.log(err);
      reject(err);
    }
  });
}; */


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
    yahooFinance.snapshot(snapshotOptions).then((result) => {
      resolve(result);
    }).catch((err) => {
      reject(err);
    });
  });
};

let processIndexResults = asyncify(function(results, symbolLookup) {
  if (results) {
    // Check if multi-dimensional
    if (Array.isArray(results)) {
      // Multiple  symbols returned
      results.forEach((indResult) => {
        awaitify(processResult(indResult, symbolLookup));
      });
    } else {
      // Single symbol returned
      awaitify(processResult(results, symbolLookup));
    }
  }
});

let processResult = asyncify(function(result, symbolLookup) {
  // Convert yahoo symbol to generic symbol
  result.symbol = symbolLookup[result.symbol];

  Object.keys(result).forEach((field) => {
    // Reset number here required
    result[field] = utils.checkForNumber(result[field]);
  });

  awaitify(writeIndexQuote(result));
});

let processCompanyResults = asyncify(function(results, symbolLookup,
  dataToAppend, dividends) {
  if (results) {
    // Check if multi-dimensional
    if (Array.isArray(results)) {
      // Multiple  symbols returned
      results.forEach((indResult) => {
        awaitify(processCompanyResult(indResult, symbolLookup,
          dataToAppend, dividends));
      });
    } else {
      // Single symbol returned
      awaitify(processCompanyResult(results, symbolLookup,
        dataToAppend, dividends));
    }
  }
});

let processCompanyResult = asyncify(function(result, symbolLookup,
  dataToAppend, dividends) {
  // Convert yahoo symbol to generic symbol
  result.symbol = symbolLookup[result.symbol];

  Object.keys(result).forEach((field) => {
    // Reset number here required
    result[field] = utils.checkForNumber(result[field]);
  });

  // Add metric data to this result
  // result = awaitify(addMetricsToQuote(result));

  // If data to append,ietrate through keys and add it to the record
  if (dataToAppend) {
    Object.keys(dataToAppend).forEach((dataKey) => {
      result[dataKey] = dataToAppend[dataKey];
    });
  }

  // Add dividends details if located
  if (dividends && dividends[result.symbol]) {
    result['exDividendDate'] = dividends[result.symbol]['exDividendDate'];
    result['exDividendPayout'] = dividends[result.symbol]['exDividendPayout'];
  }

  // Write value
  writeCompanyQuoteData(result);
});

let retrieveDividends = function(symbol, startDate, endDate) {
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

/*
let addMetricsToQuote = function(quote) {
  if (!quote['symbol']) {
    console.log('Call addMetricsToQuote missing quote.symbol element');
    return quote;
  }
  try {
    let meVal = awaitify(
      metrics.returnCompanyMetricValuesForDate(
        quote['symbol'], utils.returnDateAsString(quote['lastTradeDate'])));

    metricsFields.forEach((metricField) => {
      if (meVal[metricField] || meVal[metricField] === 0) {
        quote[metricField] = meVal[metricField];
      }
    });

  // console.log(quote);
  } catch (err) {
    console.log(err);
  }

  return quote;
};

let writeIndexResults = asyncify(function(indexDataSet) {
  console.log('----- Start write index quote results -----');

  if (indexDataSet.length) {
    indexDataSet.forEach((indexQuote) => {
      try {
        awaitify(writeIndexQuote(indexQuote));
      } catch (err) {
        console.log(err);
      }
    });
  }
}); */

let writeIndexQuote = asyncify(function(indexQuote) {
  // console.log('----- Write index quote  -----');
  try {
    // Set up the basic insert structure for dynamo
    let insertDetails = {
      tableName: 'indexQuotes',
      values: {},
      primaryKey: [
        'symbol', 'quoteDate',
      ],
    };

    indexQuote.quoteDate = utils
      .returnDateAsString(indexQuote['lastTradeDate']);
    indexQuote.yearMonth = indexQuote.quoteDate
      .substring(0, 7).replace('-', '');

    // Check through for values with null and remove from object
    Object.keys(indexQuote).forEach((field) => {
      if (indexQuote[field] === null) {
        delete indexQuote[field];
      }
    });

    insertDetails.values = indexQuote;
    awaitify(dynamodb.insertRecord(insertDetails));
  } catch (err) {
    console.log(err);
  }
});

/**
 * Converts individual index data records into an array of values which
 * can be appended to every company symbol record
 * @param {Array} indexData the data in the format:
 *    {
 *    symbol: ALLORD
 *    lastTradeDate: 2017-02-03
 *     ...
 *    }
 * @return {Array}  Array in the form of
 *    {
 *      "allordpreviousclose": 5696.4,
 *      "allordchange": -23.9,
 *      "allorddayslow": 5668.9,
 *      ...
 *    }
 */
let convertIndexDatatoAppendData = function(indexData) {
  let returnVal = {};

  indexData.forEach((indexRow) => {
    // console.log(indexRow);
    indexPrefix = indexRow['symbol'].toLowerCase();
    returnVal[indexPrefix + 'previousclose'] = indexRow['previousClose'];
    returnVal[indexPrefix + 'change'] = indexRow['change'];
    returnVal[indexPrefix + 'dayslow'] = indexRow['daysLow'];
    returnVal[indexPrefix + 'dayshigh'] = indexRow['daysHigh'];
    returnVal[indexPrefix + 'percebtChangeFrom52WeekHigh'] = indexRow['percebtChangeFrom52WeekHigh'];
    returnVal[indexPrefix + 'percentChangeFrom52WeekLow'] = indexRow['percentChangeFrom52WeekLow'];
  });

  console.log(returnVal);
  return returnVal;
};

/**
 * Retrieves index values for a specific date stored in dynamodb
 * @param {String} dateVal the date
 * @return {Array}  Array in the form of
 *    {
 *      "allordpreviousclose": 5696.4,
 *      "allordchange": -23.9,
 *      "allorddayslow": 5668.9,
 *      ...
 *    }
 */
let returnIndexDataForDate = function(dateVal) {
  if (!utils.isDate(dateVal)) {
    console.error(`Invalid dateVal: ${dateVal}`);
    return;
  }
  let returnVal = {};
  let quoteDate = utils.returnDateAsString(dateVal);
  let allIndiceResults = [];

  let queryDetails = {
    tableName: 'indexQuotes',
    keyConditionExpression: 'symbol = :symbol and quoteDate <= :quoteDate',
    expressionAttributeValues: {
      ':quoteDate': quoteDate,
    },
    reverseOrder: true,
    limit: 1,
  };


  let symbolResult = awaitify(setupSymbols(true));
  let indexSymbols = symbolResult.indexSymbols;

  indexSymbols.forEach((index) => {
    queryDetails.expressionAttributeValues[':symbol'] = index;
    let indexResults = awaitify(dynamodb.queryTable(queryDetails));
    if (indexResults.length) {
      allIndiceResults = allIndiceResults.concat(indexResults);
    }
  });


  // Check we got a metrics result
  if (allIndiceResults.length) {
    returnVal = convertIndexDatatoAppendData(allIndiceResults);
  }

  return returnVal;
};


/**
 * Copies properties from two objects to a new object
 * @param {Object} object1
 * @param {Object} object2
 * @return {Object}  a new object with all the properties
 */
let addObjectProperties = function(object1, object2) {
  let returnObj = {};

  Object.keys(object1).forEach((key) => {
    returnObj[key] = object1[key];
  });

  Object.keys(object2).forEach((key) => {
    returnObj[key] = object2[key];
  });

  console.log('----- Appended object -----');
  console.log(JSON.stringify(returnObj));

  return returnObj;
};

/**
 * Appends data to every compamy row
 * @param {Object} dataVals the data in the format:
 *    {
 *    data: base data array
 *    fields: fields list array
 *    append: object with key/value pairs to append to every row
 *    }
 * @return {Object}  Object in the form of
 *    {
 *    data: updated data array
 *    fields: updated fields list array
 *    }
 */
/* let addAppendDataToRows = function(dataVals) {
  let workingData = dataVals;
  // Check if data is present to append
  if (!workingData.append) {
    return workingData;
  }

  // Check if fields have been supplied
  if (workingData.fields) {
    // Work through array and retrieve the key of each value
    Object.keys(workingData.append).forEach((appendKey) => {
      if (workingData.fields.indexOf(appendKey) < 0) {
        workingData.fields.push(appendKey);
      }
    });
  }

  // Append data to every row
  for (c = 0; c < workingData.data.length; c++) {
    Object.keys(workingData.append).forEach((appendKey) => {
      workingData.data[c][appendKey] = workingData.append[appendKey];
    });
  }

  return workingData;
}; */

/**
 * Appends data to every compamy row
 * @param {Object}  fieldData data in the format:
 *    {
 *    fields: base data array
 *    appendFields: fields list array
 *    }
 * @return {Array}  Updated field list
 */
/* let appendDataFieldNamesToDataSet = function(fieldData) {
  // Check if fields have been supplied
  if (fieldData.fields && fieldData.appendFields) {
    // Work through array and retrieve the key of each value
    Object.keys(fieldData.appendFields).forEach((appendKey) => {
      if (fieldData.fields.indexOf(appendKey) < 0) {
        fieldData.fields.push(appendKey);
      }
    });
  }

  return fieldData.fields;
}; */

/**
 * Looks up current company Metrics and appends them to each company row
 * @param {Object} dataVals the data in the format:
 *    {
 *    data: base data array
 *    fields: fields list array
 *    }
 * @return {Object}  Object in the form of
 *    {
 *    data: updated data array
 *    fields: updated fields list array
 *    }
 */
/* let addMetricDataToRows = asyncify(function(dataVals) {
  return new Promise(function(resolve, reject) {
    try {
      let wkData = dataVals;

      // Check if fields have been supplied
      if (wkData.fields) {
        // Make sure the metrics fields are added to the fields list
        metricsFields.forEach((field) => {
          if (wkData.fields.indexOf(field) === -1) {
            wkData.fields.push(field);
          }
        });
      }

      // Append data to every row
      for (let c = 0; c < wkData.data.length; c++) {
        let companSymbol = wkData.data[c]['symbol'];

        awaitify(
          metrics.returnCompanyMetricValuesForDate(
            companSymbol, utils.returnDateAsString(Date.now()))
            .then((meVal) => {
              if (meVal['symbol']) {
                metricsFields.forEach((metricField) => {
                  if (meVal[metricField] || meVal[metricField] === 0) {
                    wkData.data[c][metricField] = meVal[metricField];
                  }
                });
              }
              console.log(wkData.data[c]);
            })
            .catch((err) => {
              console.log(err);
            })
        );
      }

      resolve(wkData);
    } catch (err) {
      reject(err);
    }
  });
});

let addMetricsFieldNamesToDataSet = function(dataset) {
  if (!dataset.Fields) {
    console.log('Call addMetricsFieldNames missing dataset.fields.');
    return dataset;
  }
  try {
    // Check if fields have been supplied
    if (dataset.fields) {
      // Make sure the metrics fields are added to the fields list
      metricsFields.forEach((field) => {
        if (dataset.fields.indexOf(field) === -1) {
          dataset.fields.push(field);
        }
      });
    }
  } catch (err) {
    console.log(err);
  }
  return dataset;
};

let writeCompanyQuoteDataSet = asyncify(function(quoteObject, quoteDate) {
  let insertDetails = {
    tableName: 'companyQuotes',
    values: {},
    primaryKey: [
      'symbol', 'quoteDate',
    ],
  };

  // Iterate through quotes and save each one
  quoteObject.forEach((quote) => {
    quote.quoteDate = quoteDate;
    quote.yearMonth = quoteDate.substring(0, 7).replace('-', '');

    // Check through for values with null and remove from object
    Object.keys(quote).forEach((field) => {
      if (quote[field] === null) {
        delete quote[field];
      }
    });

    insertDetails.values = quote;
    awaitify(dynamodb.insertRecord(insertDetails));
  });
});*/

let writeCompanyQuoteData = asyncify(function(quoteData) {
  // If unexpected r3cords come back with no trade data, skop them
  if (!quoteData['lastTradeDate']) {
    return;
  }

  let insertDetails = {
    tableName: 'companyQuotes',
    values: {},
    primaryKey: [
      'symbol', 'quoteDate',
    ],
  };

  quoteData.quoteDate = utils.returnDateAsString(quoteData['lastTradeDate']);
  quoteData.yearMonth = quoteData['quoteDate'].substring(0, 7).replace('-', '');

  delete quoteData['lastTradeDate'];

  // Check through for values with null and remove from object
  Object.keys(quoteData).forEach((field) => {
    if (quoteData[field] === null || quoteData[field] === '') {
      delete quoteData[field];
    }
  });

  insertDetails.values = quoteData;
  awaitify(dynamodb.insertRecord(insertDetails));
});

let updateQuotesWithMetrics = asyncify(function(symbols, quoteDate, recLimits) {
  if (!symbols) {
    console.log('updateQuotesWithMetrics error: no symbols supplied');
    return;
  }

  if (!quoteDate) {
    console.log('updateQuotesWithMetrics error: no quoteDate supplied');
    return;
  }

  let startRec = 0;
  let endRec = symbols.length - 1;

  if (recLimits && recLimits.startRec) {
    startRec = recLimits.startRec;
  }

  if (recLimits && recLimits.endRec && recLimits.endRec < endRec) {
    endRec = recLimits.endRec;
  }

  if (startRec > endRec) {
    return;
  }

  console.log('Updating company quotes with metrics for ', quoteDate, ' from ',
    startRec, ' to ', endRec);

  let metricsDate = quoteDate;

  // Set-up update details, only update the record if it is found
  let updateDetails = {
    tableName: 'companyQuotes',
    key: {
      quoteDate: metricsDate,
    },
    conditionExpression: 'attribute_exists(symbol) and ' +
      'attribute_exists(quoteDate)',
  };

  for (let c = startRec; c <= endRec; c++) {
    let companySymbol = symbols[c];
    // symbols.forEach((companySymbol) => {
    // If at least one quote record has been returned, continue
    let fieldsPresent = [];
    let updateExpression;
    let expressionAttributeValues = {};
    let expressionAttributeNames = {};

    // Set-up query and retrieve metrics value
    let queryDetails = {
      tableName: 'companyMetrics',
      keyConditionExpression: 'symbol = :symbol and ' +
        'metricsDate <= :metricsDate',
      expressionAttributeValues: {
        ':metricsDate': metricsDate,
        ':symbol': companySymbol,
      },
      reverseOrder: true,
      limit: 1,
    };

    let metricsResult = awaitify(dynamodb.queryTable(queryDetails));

    // Check we got a metrics result
    if (metricsResult.length) {
      let workingRecord = metricsResult[0];

      // Set up the key: symbol and quoteDate
      updateDetails.key.symbol = companySymbol;

      // Remove the attributes which should not be copied into cpmanyQuotes
      delete workingRecord['symbol'];
      delete workingRecord['metricsDate'];
      delete workingRecord['yearMonth'];
      delete workingRecord['created'];

      // Get a list of fields and values to copy
      Object.keys(workingRecord).forEach((field) => {
        expressionAttributeValues[(':' + field)] = workingRecord[field];
        expressionAttributeNames[('#' + field)] = field;
        fieldsPresent.push('#' + field + '=:' + field);
      });

      // Enure that at least one field is present to update
      if (fieldsPresent.length) {
        updateExpression = 'set ' + fieldsPresent.join(',');

        updateDetails.updateExpression = updateExpression;
        updateDetails.expressionAttributeValues = expressionAttributeValues;
        updateDetails.expressionAttributeNames = expressionAttributeNames;

        try {
          awaitify(dynamodb.updateRecord(updateDetails));
        } catch (err) {
          console.log(err);
        }
      }
    }
  }
// });
});

/* let updateQuotesWithMetricsOld = asyncify(function(symbols, quoteDate,
  searchForward) {
  if (!symbols) {
    console.log('updateQuotesWithMetrics error: no symbols supplied');
    return;
  }

  if (!quoteDate) {
    console.log('updateQuotesWithMetrics error: no quoteDate supplied');
    return;
  }

  console.log('Updating company quotes with metrics for ', quoteDate);

  let metricsDate = quoteDate;

  // Set-up update details, only update the record if it is found
  let updateDetails = {
    tableName: 'companyQuotes',
  };

  symbols.forEach((companySymbol) => {
    // Confirm record(s) exist to update
    let queryQuotes = {
      tableName: 'companyQuotes',
      filterExpression: 'attribute_not_exists(EPS) and ' +
        'attribute_not_exists(BookValuePerShareYear) and ' +
        'attribute_not_exists(MarketCap)',
      expressionAttributeValues: {
        ':symbol': companySymbol,
        ':quoteDate': quoteDate,
      },
      projectionExpression: 'symbol, quoteDate',
    };

    if (searchForward) {
      // Need to search forwards for any records on or after selected
      //    dateParameter which are missing metrics info
      queryQuotes.keyConditionExpression = 'symbol = :symbol and ' +
        ' quoteDate >= :quoteDate';
    } else {
      // only look for quotes on the selected day
      queryQuotes.keyConditionExpression = 'symbol = :symbol and ' +
        ' quoteDate = :quoteDate';
    }

    let qResult = awaitify(dynamodb.queryTable(queryQuotes));

    if (qResult.length) {
      // If at least one quote record has been returned, continue
      let fieldsPresent = [];
      let updateExpression;
      let expressionAttributeValues = {};
      let expressionAttributeNames = {};
      let quoteDates = [];

      // Set-up date(s) to update for this company
      qResult.forEach((result) => {
        quoteDates.push(result['quoteDate']);
      });


      // Set-up query and retrieve metrics value
      let queryDetails = {
        tableName: 'companyMetrics',
        keyConditionExpression: 'symbol = :symbol and ' +
          'metricsDate = :metricsDate',
        expressionAttributeValues: {
          ':metricsDate': metricsDate,
          ':symbol': companySymbol,
        },
        limit: 1,
      };

      let metricsResult = awaitify(dynamodb.queryTable(queryDetails));

      // Check we got a metrics result
      if (metricsResult.length) {
        let workingRecord = metricsResult[0];

        // Work through quote dates to execute update for
        quoteDates.forEach((qDate) => {
          // Set up the key: symbol and quoteDate
          updateDetails.key = {
            symbol: companySymbol,
            quoteDate: qDate,
          };

          // Remove the attributes which should not be copied into cpmanyQuotes
          delete workingRecord['symbol'];
          delete workingRecord['metricsDate'];
          delete workingRecord['yearMonth'];
          delete workingRecord['created'];

          // Get a list of fields and values to copy
          Object.keys(workingRecord).forEach((field) => {
            expressionAttributeValues[(':' + field)] = workingRecord[field];
            expressionAttributeNames[('#' + field)] = field;
            fieldsPresent.push('#' + field + '=:' + field);
          });

          // Enure that at least one field is present to update
          if (fieldsPresent.length) {
            updateExpression = 'set ' + fieldsPresent.join(',');

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
      }
    }
  });
}); */

let getCurrentExecutionDate = asyncify(function() {
  // Get current dat based on last index quote for All Ordinaries
  let queryDetails = {
    tableName: 'indexQuotes',
    keyConditionExpression: 'symbol = :symbol',
    reverseOrder: true,
    expressionAttributeValues: {
      ':symbol': 'ALLORD',
    },
    limit: 1,
    projectionExpression: 'symbol, quoteDate',
  };

  let indexLookup = awaitify(dynamodb.queryTable(queryDetails));
  if (indexLookup.length) {
    return indexLookup[0]['quoteDate'];
  } else {
    return null;
  }
});

let executeFinancialIndicators = asyncify(function() {
  awaitify(finIndicators.updateIndicatorValues());
});

let executeCompanyMetrics = asyncify(function() {
  let symbolResult = awaitify(setupSymbols());
  let mCompanies = symbolResult.companies;

  awaitify(metrics.updateCompanyMetrics(mCompanies));
});


let executeIndexQuoteRetrieval = asyncify(function() {
  let symbolResult = awaitify(setupSymbols(true));

  let symbolLookup = symbolResult.symbolLookup;
  let indices = symbolResult.indices;


  console.log('----- Start retrieve index quotes -----');
  try {
    let results = awaitify(retrieveSnapshot(indices, indiceFieldsToRetrieve));
    awaitify(processIndexResults(results, symbolLookup));
  } catch (err) {
    console.log(err);
  }
});

let executeCompanyQuoteRetrieval = asyncify(function() {
  let dataToAppend = {};
  let symbolGroups = [];
  let symbolResult = awaitify(setupSymbols());

  let symbolLookup = symbolResult.symbolLookup;
  let companies = symbolResult.companies;

  let todayString = utils.returnDateAsString(Date.now());
  let financialIndicatos = awaitify(finIndicators
    .returnIndicatorValuesForDate(todayString));
  let indexDataToAppend = awaitify(returnIndexDataForDate(todayString));

  // create append array from indicators and index data
  dataToAppend = addObjectProperties(financialIndicatos,
    indexDataToAppend);

  console.log('----- Start retrieve company quotes -----');

  /* Split companies into groups of 15 to ensure request doesn't exceed api
      url length */
  for (companyCounter = 0; companyCounter < companies.length;
    companyCounter += 15) {
    symbolGroups.push(companies.slice(companyCounter, companyCounter + 15));
  }

  symbolGroups.forEach((symbolGroup) => {
    try {
      let result = awaitify(retrieveSnapshot(symbolGroup,
        companyFieldsToRetrieve));

      // Retrieve dividends
      let dividends = awaitify(getDividendsforDate(symbolGroup,
        todayString, symbolLookup));

      awaitify(processCompanyResults(result, symbolLookup,
        dataToAppend, dividends));
    } catch (err) {
      console.log(err);
    }
  });
});

let getDividendsforDate = asyncify(function(symbolGroup, retrievalDate,
  symbolLookup) {
  if (!symbolGroup) {
    console.error('getDividendsforDate error: no symbolGroup supplied');
    return;
  }

  if (!retrievalDate) {
    console.error('getDividendsforDate error: no retrievalDate supplied');
    return;
  }

  if (!symbolLookup) {
    console.error('getDividendsforDate error: no symbolLookup supplied');
    return;
  }


  // Retrieve dividends
  dividendEndDate = utils.returnDateAsString(retrievalDate);
  dividendStartDate = utils.dateAdd(dividendEndDate, 'years', -1);
  dividendStartDate = utils.dateAdd(dividendStartDate, 'days', 1);
  let dividends = awaitify(retrieveDividends(symbolGroup,
    dividendStartDate, dividendEndDate));

  // Convert dividend results array to results object
  let dividendResults = {};

  // Single company result is returned as an array
  if (Array.isArray(dividends) && dividends.length > 0) {
    // Multiple  symbols returned
    dividends.forEach((dividendRecord) => {
      processDividend(dividendRecord);
    });
  } else {
    // Multi-company result is returned as an object
    Object.keys(dividends).forEach((dividendCompany) => {
      dividends[dividendCompany].forEach((dividendRecord) => {
        processDividend(dividendRecord);
      });
    });
  }

  /** Check if dividend record is the latest record for each symbol.  If it is
  *    pdate the recorded dividend
  * @param {Object} dividendRecord record to process
  */
  function processDividend(dividendRecord) {
    let symbol = symbolLookup[dividendRecord.symbol];
    let exDividendDate = utils.returnDateAsString(dividendRecord['date']);
    let exDividendPayout = dividendRecord['dividends'];

    /* Check if symbol has already been recorded, or recorded with an earlier
      date */
    if (!dividendResults[symbol] ||
      dividendResults[symbol]['exDividendDate'] < exDividendDate)
      dividendResults[symbol] = {};
    dividendResults[symbol]['exDividendDate'] = exDividendDate;
    dividendResults[symbol]['exDividendPayout'] = exDividendPayout;
  }

  return dividendResults;
});

let executeMetricsUpdate = asyncify(function(recLimits) {
  // Get current dat based on last index quote for All Ordinaries
  let metricsDate = awaitify(getCurrentExecutionDate());
  let companies = [];
  let symbolResult = awaitify(setupSymbols());

  Object.keys(symbolResult.companyLookup).forEach((yahooSymbol) => {
    companies.push(symbolResult.companyLookup[yahooSymbol]);
  });

  if (metricsDate) {
    awaitify(updateQuotesWithMetrics(companies, metricsDate, recLimits || {}));
  }
});

let executeAll = asyncify(function() {
  let t0 = new Date();

  console.log('Executing retrieve financial indicators');
  awaitify(executeFinancialIndicators());
  let tf = new Date();

  console.log('Executing retrieve company metrics');
  awaitify(executeCompanyMetrics());
  let tm = new Date();

  console.log('Executing retrieve index quotes');
  awaitify(executeIndexQuoteRetrieval());
  let ti = new Date();

  console.log('Executing retrieve company quotes');
  awaitify(executeCompanyQuoteRetrieval());
  let tc = new Date();

  console.log('Executing update company quotes with metrics - phase 1');
  awaitify(executeMetricsUpdate({
    startRec: 0,
    endRec: 999,
  }));
  let tu = new Date();

  console.log('Executing update company quotes with metrics - phase 2');
  awaitify(executeMetricsUpdate({
    startRec: 1000,
    endRec: 1999,
  }));
  let tu2 = new Date();

  console.log('Executing update company quotes with metrics - phase 3');
  awaitify(executeMetricsUpdate({
    startRec: 2000,
  }));
  let tu3 = new Date();


  console.log('--------- All done --------');

  console.log('Retrieve financial indicators: ',
    utils.dateDiff(t0, tf, 'seconds'), ' seconds to execute.');

  console.log('Retrieve company metrics: ',
    utils.dateDiff(tf, tm, 'seconds'), ' seconds to execute.');

  console.log('Retrieve index quotes: ',
    utils.dateDiff(tm, ti, 'seconds'), ' seconds to execute.');

  console.log('Retrieve company quotes: ',
    utils.dateDiff(ti, tc, 'seconds'), ' seconds to execute.');

  console.log('Update company quotes with metrics data - phase 1: ',
    utils.dateDiff(tq, tu, 'seconds'), ' seconds to execute.');

  console.log('Update company quotes with metrics data - phase 2: ',
    utils.dateDiff(ttu, tu2, 'seconds'), ' seconds to execute.');

  console.log('Update company quotes with metrics data - phase 3: ',
    utils.dateDiff(tu2, tu3, 'seconds'), ' seconds to execute.');


  console.log('Total time for all operations: ',
    utils.dateDiff(t0, tu, 'seconds'), ' seconds to execute.');
});

let executeCommand = asyncify(function() {
  program
    .version('0.0.1')
    .description('Sharecast share data retrieval')
    .option('-f, --financial', 'Retrieve financial indicators')
    .option('-m, --metrics', 'Retrieve company metrics')
    .option('-i, --indices', 'Retrieve index quotes')
    .option('-c, --companies', 'Retrieve company quotes')
    .option('-u, --updates', 'Update company quotes with metrics data')
    .option('-a, --all', 'Perform complete series of operations')
    .parse(process.argv);

  let t0 = new Date();

  if (program.financial) {
    console.log('Executing retrieve financial indicators');
    awaitify(executeFinancialIndicators());
    let tf = new Date();
    console.log('Retrieve financial indicators: ',
      utils.dateDiff(t0, tf, 'seconds'), ' seconds to execute.');
  }
  if (program.metrics) {
    console.log('Executing retrieve company metrics');
    awaitify(executeCompanyMetrics());
    let tm = new Date();
    console.log('Retrieve company metrics: ',
      utils.dateDiff(t0, tm, 'seconds'), ' seconds to execute.');
  }
  if (program.indices) {
    console.log('Executing retrieve index quotes');
    awaitify(executeIndexQuoteRetrieval());
    let ti = new Date();
    console.log('Retrieve index quotes: ',
      utils.dateDiff(t0, ti, 'seconds'), ' seconds to execute.');
  }
  if (program.companies) {
    console.log('Executing retrieve company quotes');
    awaitify(executeCompanyQuoteRetrieval());
    let tc = new Date();
    console.log('Retrieve company quotes: ',
      utils.dateDiff(t0, tc, 'seconds'), ' seconds to execute.');
  }
  if (program.updates) {
    console.log('Executing update company quotes with metrics');
    awaitify(executeMetricsUpdate());
    let tu = new Date();
    console.log('Update company quotes with metrics data: ',
      utils.dateDiff(t0, tu, 'seconds'), ' seconds to execute.');
  }
  if (program.all) {
    console.log('Executing complete series');
    executeAll();
  }
});

executeCommand();

/* let testDividends = asyncify(function() {
  let symbolResult = awaitify(setupSymbols());

  symbolLookup = symbolResult.symbolLookup;
  indexLookup = symbolResult.indexLookup;
  indexSymbols = symbolResult.indexSymbols;
  companyLookup = symbolResult.companyLookup;
  indices = symbolResult.indices;
  companies = symbolResult.companies;

  let todayString = utils.returnDateAsString(Date.now());

  let symbolGroups = [];

  for (companyCounter = 0; companyCounter < companies.length;
    companyCounter += 15) {
    symbolGroups.push(companies.slice(companyCounter, companyCounter + 15));
  }

  symbolGroups.forEach((symbolGroup) => {
    try {
      // Retrieve dividends
      let dividends = awaitify(getDividendsforDate(symbolGroup,
        todayString, symbolLookup));

      console.log(dividends);
    } catch (err) {
      console.error(err);
    }
  });
});

testDividends(); */

module.exports = {
  setupSymbols: setupSymbols,
  updateQuotesWithMetrics: updateQuotesWithMetrics,
  writeIndexQuote: writeIndexQuote,
  writeCompanyQuoteData: writeCompanyQuoteData,
  returnIndexDataForDate: returnIndexDataForDate,
};
