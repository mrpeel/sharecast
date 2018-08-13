'use strict';

const retrieveQuote = require('./retrieve-quote');
const utils = require('./utils');
const dynamodb = require('./dynamodb');
const symbols = require('./dynamo-symbols');
const finIndicators = require('./dynamo-retrieve-financial-indicator-data');

/**  Retrieve company and index symbols and sets them up for retrieval
 * @param {Boolean} indicesOnly - whether to only return the index symbols
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
let setupSymbols = async function(indicesOnly) {
  try {
    console.log('----- Start setup symbols -----');
    let indexValues = await symbols.getIndices();
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
      let companyValues = await symbols.getCompanies();

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
};


/**  Process an array of index quote results
 * @param {Array} results - an array with index quote values
 *                         look up or  a single index quote value
 * @param {Array} symbolLookup - an array of key/value pairs to translate
 *                               the symbols yahoo uses to the normal symbol
 */
let processIndexResults = async function(results, symbolLookup) {
  // Work through results and add them to db
  let resultSymbols = Object.keys(results);
  let errorCount = 0;
  let insertResults = {};

  for (const resultSymbol of resultSymbols) {
    try {
      let result = results[resultSymbol];
      // Convert yahoo symbol to generic symbol
      result.symbol = symbolLookup[result.symbol];

      // Set up the basic insert structure for dynamo
      let insertDetails = {
        tableName: 'indexQuotes',
        values: {},
        primaryKey: [
          'symbol', 'quoteDate',
        ],
      };

      result.quoteDate = utils
        .returnDateAsString(result['lastTradeDate']);
      result.yearMonth = result.quoteDate
        .substring(0, 7).replace('-', '');

      // Check through for values with null and remove from object
      Object.keys(result).forEach((field) => {
        if (result[field] === null ||
          result[field] === 'Infinity' ||
          result[field] === '-Infinity' ||
          Number.isNaN(result[field])) {
          delete result[field];
        }
      });

      insertDetails.values = result;
      let writeResult = await dynamodb.insertRecord(insertDetails);

      // Create result type if doesn't exist
      if (!insertResults[writeResult.result]) {
        insertResults[writeResult.result] = 0;
      }
      // Increment result type
      insertResults[writeResult.result]++;
    } catch (err) {
      console.log(err);
      errorCount++;
    }
  }

  return {
    insertResults: insertResults,
    errorCount: errorCount,
  };
};


/**  Process an array of company quote results
 * @param {Object} results - an object with symbol quote values
 *                         look up or  a single index quote value
 * @param {Array} symbolLookup - an array of key/value pairs to translate
 *                               the symbols yahoo uses to the normal symbol
 * @param {Object} dataToAppend - an object with a set of data properties to
 *                                append to all symbol records
 * @param {Object} dividends - An object with symbol dividends to append
 *                              to symbol records when saving
 */
let processCompanyResults = async function(results, symbolLookup,
  dataToAppend, dividends) {
  let symbols = Object.keys(results) || [];
  let errorCount = 0;
  let insertResults = {};

  for (const symbol of symbols) {
    try {
      let result = results[symbol];

      // Convert yahoo symbol to generic symbol
      result.symbol = symbolLookup[result.symbol];

      Object.keys(result).forEach((field) => {
        // Reset number here if required
        if (result[field]) {
          result[field] = utils.checkForNumber(result[field]);
        }
      });

      // If data to append,ietrate through keys and add it to the record
      if (dataToAppend) {
        Object.keys(dataToAppend).forEach((dataKey) => {
          result[dataKey] = dataToAppend[dataKey];
        });
      }

      // Add dividends details if located
      if (dividends && dividends[symbol]) {
        result['exDividendDate'] = dividends[symbol]['exDividendDate'];
        result['exDividendPayout'] = dividends[symbol]['exDividendPayout'];
      }

      // Write value
      let writeResult = await writeCompanyQuoteData(result);

      // Create result type if doesn't exist
      if (!insertResults[writeResult.result]) {
        insertResults[writeResult.result] = 0;
      }
      // Increment result type
      insertResults[writeResult.result]++;
    } catch (err) {
      console.log(err);
      errorCount++;
    }
  }

  return {
    insertResults: insertResults,
    errorCount: errorCount,
  };
};


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
    let indexPrefix = indexRow['symbol'].toLowerCase();
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
let returnIndexDataForDate = async function(dateVal) {
  if (!utils.isDate(dateVal)) {
    console.error(`Invalid dateVal: ${dateVal}`);
    return;
  }
  let returnVal = {};
  let quoteDate = utils.returnDateAsString(dateVal);
  let allIndexResults = [];

  let queryDetails = {
    tableName: 'indexQuotes',
    keyConditionExpression: 'symbol = :symbol and quoteDate <= :quoteDate',
    expressionAttributeValues: {
      ':quoteDate': quoteDate,
    },
    reverseOrder: true,
    limit: 1,
  };


  let symbolResult = await setupSymbols(true);
  let indexSymbols = symbolResult.indexSymbols;

  for (const index of indexSymbols) {
    queryDetails.expressionAttributeValues[':symbol'] = index;
    let indexResults = await dynamodb.queryTable(queryDetails);
    if (indexResults.length) {
      allIndexResults = allIndexResults.concat(indexResults);
    }
  }


  // Check we got a metrics result
  if (allIndexResults.length) {
    returnVal = convertIndexDatatoAppendData(allIndexResults);
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

/**  Write a conpany quote record to dynamodb.
 *    Converts lastTradeDate -> quoteDate, copies lastTradePriceOnly ->
 *     adjustedPrice, and checks and removes any invalid values.
 * @param {Object} quoteData - the company quote to write
 */
let writeCompanyQuoteData = async function(quoteData) {
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
  if (!quoteData.adjustedPrice) {
    quoteData.adjustedPrice = quoteData['lastTradePriceOnly'];
  }

  delete quoteData['lastTradeDate'];

  // Check through for values with null and remove from object
  Object.keys(quoteData).forEach((field) => {
    if (quoteData[field] === null ||
      quoteData[field] === 'Infinity' ||
      quoteData[field] === '-Infinity' ||
      Number.isNaN(quoteData[field])) {
      delete quoteData[field];
    }
  });

  insertDetails.values = quoteData;
  return await dynamodb.insertRecord(insertDetails);
};


/**  Executes the retrieval and storage of the latest financial indicator
 *     information
 */
let executeFinancialIndicators = async function() {
  return await finIndicators.updateIndicatorValues();
};


/**  Executes the retrieval and storage of the latest index quote
 *     information
 */
let executeIndexQuoteRetrieval = async function() {
  let symbolResult = await setupSymbols(true);

  let symbolLookup = symbolResult.symbolLookup;
  let indices = symbolResult.indices;

  console.log('----- Start retrieve index quotes -----');
  try {
    let indexResults = await retrieveQuote.getQuotes({
      'symbols': indices,
      'type': 'index',
    });

    console.log(`Number of succesfull results: ${Object.keys(indexResults.results).length}`);
    console.log(`Number of errors: ${indexResults.errorCount}`);

    let processResults;

    if (indexResults.results) {
      processResults = await processIndexResults(indexResults.results, symbolLookup);
    }
    return {
      indexRetrieves: Object.keys(indexResults.results).length,
      indexRetrieveErrors: indexResults.errorCount,
      insertQuoteProcessingResults: processResults,
    };
  } catch (err) {
    console.log(err);
    return {
      error: err,
    };
  }
};

/**  Executes the retrieval and storage of the latest company quote
*     information
* @param {Object} recLimits   (optional) - specifies where to start and stop
*     the operation, in form of:
*     {
        startRec: 100,
        endRec, 199,
*      }
*     This would process from company record index 100 to 199
*/
let executeCompanyQuoteRetrieval = async function(recLimits) {
  try {
    let dataToAppend = {};
    let symbolResult = await setupSymbols();

    let symbolLookup = symbolResult.symbolLookup;
    let companies = symbolResult.companies;

    let todayString = utils.returnDateAsString(Date.now());
    let financialIndicatos = await finIndicators.returnIndicatorValuesForDate(todayString);
    let indexDataToAppend = await returnIndexDataForDate(todayString);

    // create append array from indicators and index data
    dataToAppend = addObjectProperties(financialIndicatos,
      indexDataToAppend);

    let startRec = 0;
    let endRec = companies.length - 1;

    if (recLimits && recLimits.startRec) {
      startRec = recLimits.startRec;
    }

    if (recLimits && recLimits.endRec && recLimits.endRec < endRec) {
      endRec = recLimits.endRec;
    }

    if (startRec > endRec) {
      return;
    }

    console.log('Retrieving company quotes from ', startRec, ' to ', endRec);

    // Get array of companies to work with
    let companySymbols = companies.slice(startRec, endRec);
    let companyResults = await retrieveQuote.getQuotes({
      'symbols': companySymbols,
      'type': 'company',
    });

    console.log(`Number of succesfull quote results: ${Object.keys(companyResults.results).length}`);
    console.log(`Number of quote errors: ${companyResults.errorCount}`);

    let dividendEndDate = utils.returnDateAsString(todayString);
    let dividendStartDate = utils.dateAdd(todayString, 'years', -1);
    dividendStartDate = utils.dateAdd(dividendStartDate, 'days', 1);

    let dividendResults = await retrieveQuote.getLatestDividends({
      'symbols': companySymbols,
      'startDate': dividendStartDate,
      'endDate': dividendEndDate,
    });

    console.log(`Number of succesfull dividend results:`,
      `${JSON.stringify(Object.keys(dividendResults.results).length)}`);
    console.log(`Number of dividend errors: ${dividendResults.errorCount}`);

    let processResults;

    if (companyResults.results) {
      console.log('Processing results');
      processResults = await processCompanyResults(companyResults.results, symbolLookup,
        dataToAppend, dividendResults.results || {});
    }

    return {
      quoteRetrieves: Object.keys(companyResults.results).length,
      quoteRetrieveErrors: companyResults.errorCount,
      latestDividendRetrieves: Object.keys(dividendResults.results).length,
      latestDividendErrors: dividendResults,
      insertQuoteProcessingResults: processResults,
    };
  } catch (err) {
    console.log(err);
    return {
      error: err,
    };
  }
};


module.exports = {
  setupSymbols: setupSymbols,
  writeCompanyQuoteData: writeCompanyQuoteData,
  returnIndexDataForDate: returnIndexDataForDate,
  executeFinancialIndicators: executeFinancialIndicators,
  executeIndexQuoteRetrieval: executeIndexQuoteRetrieval,
  executeCompanyQuoteRetrieval: executeCompanyQuoteRetrieval,
};

let localExecute = async function() {
  dynamodb.setLocalAccessConfig();
  let results = await executeCompanyQuoteRetrieval({
    startRec: 0,
    endRec: 20,
  });
  console.log(JSON.stringify(results, null, 2));
};
localExecute();
