'use strict';

const fetch = require('node-fetch');
const utils = require('./utils');
const credentials = require('../credentials/credentials.json');
const apiKey = credentials['quandl-api-key'];
const dynamodb = require('./dynamodb');

let baseUrl = 'https://www.quandl.com/api/v3/datasets/';
let urlSuffix = '?api_key=' + apiKey;
let defaultStartDate = '2006-02-01';
let dateParameter = '&start_date=';

/* For each indicator, record the url, column to find data and time period
    for the lag of when data is available
  ----
  Column 0 is always the date of the record
*/
let indicatorRetrievalValues = {
  'FIRMMCRT': {
    /* cash rate (previous month) */
    url: 'RBA/F01_1_FIRMMCRT.json',
    dataColumn: 1,
    monthLag: 0,
    dayLag: 14,
  },
  'FXRUSD': {
    /* Daily exchange rates */
    url: 'RBA/FXRUSD.json',
    dataColumn: 1,
    monthLag: 0,
    dayLag: 0,
  },
  'GRCPAIAD': {
    /* Commodities all */
    url: 'RBA/I02.json',
    dataColumn: 1,
    monthLag: 1,
    dayLag: 1,
  },
  'GRCPRCAD': {
    /* Commodities rural */
    url: 'RBA/I02.json',
    dataColumn: 4,
    monthLag: 1,
    dayLag: 1,
  },
  'GRCPNRAD': {
    /* Commodities non-rural */
    url: 'RBA/I02.json',
    dataColumn: 7,
    monthLag: 1,
    dayLag: 1,
  },
  'GRCPBMAD': {
    /* Commodities base metals */
    url: 'RBA/I02.json',
    dataColumn: 10,
    monthLag: 1,
    dayLag: 1,
  },
  'GRCPBCAD': {
    /* Commodities non-rural bulk export */
    url: 'RBA/I02.json',
    dataColumn: 13,
    monthLag: 1,
    dayLag: 1,
  },
  'GRCPAISAD': {
    /* Commodities ]all bulk */
    url: 'RBA/I02.json',
    dataColumn: 16,
    monthLag: 1,
    dayLag: 1,
  },
  'GRCPBCSAD': {
    /* Commodities non-rural bulk */
    url: 'RBA/I02.json',
    dataColumn: 19,
    monthLag: 1,
    dayLag: 1,
  },
  'H01_GGDPCVGDP': {
    /* Aus GDP quarterly */
    url: 'RBA/H01_GGDPCVGDP.json',
    dataColumn: 1,
    monthLag: 3,
    dayLag: 2,
  },
  'H01_GGDPCVGDPFY': {
    /* Aus GDP per capita growth quarterly */
    url: 'RBA/H01_GGDPCVGDPFY.json',
    dataColumn: 1,
    monthLag: 3,
    dayLag: 2,
  },
  '640106_A3597525W': {
    /* Australia CPI (inflation) monthly */
    url: 'AUSBS/640106_A3597525W.json',
    dataColumn: 1,
    monthLag: 1,
    dayLag: 1,
  },
  'H05_GLFSEPTPOP': {
    /* Percentage of population employed */
    url: 'RBA/H05_GLFSEPTPOP.json',
    dataColumn: 1,
    monthLag: 1,
    dayLag: 1,
  },
};

/**
 * Returns an indicator value if one exists on/after the supplied start date
 * @param {Object} indicatorDetails details to retrieve:
 *  {
 *    indicatorId: indicator,
 *    url: retrievalUrl
 *    dataColumn: column number
 *  }
 * @return {Array}  An array of objects in form of:
 *    [{
 *      indicatorId: indicatorId,
 *      value-date: startDate,
 *      value: value
 *    }]
 */
let retrieveQuandlIndicator = function(indicatorDetails) {
  return new Promise(function(resolve, reject) {
    let indicatorData = [];

    fetch(indicatorDetails.url)
      .then((response) => {
        return response.json();
      })
      .then((responseJson) => {
        // console.log(responseJson);
        responseJson.dataset.data.forEach((row) => {
          indicatorData.push({
            'symbol': indicatorDetails.symbol,
            'valueDate': row[0],
            'value': row[indicatorDetails.dataColumn],
          });
        });

        resolve(indicatorData);
      })
      .catch((err) => {
        reject(err);
      });
  });
};

/**
 * Returns a list of financial indicators
 * @return {Array}  list of indicator codes to retrieve
 */
let getIndicatorList = function() {
  return Object.keys(indicatorRetrievalValues);
};

/**
 * Returns a list of financial indicators
 * @param {Array} indicatorIds list of indicator codes to retrieve
 * @return {Object}  Object in form of:
 *    {
 *      indicator: indicatorId,
 *      start-date: startDate,
 *    }
 */
let getIndicatorRequestDates = async function(indicatorIds) {
  if (!indicatorIds) {
    throw new Error('No indicators supplied: ');
  }

  try {
    let indicatorDates = [];

    // Set basic query parameters
    let queryDetails = {
      tableName: 'financialIndicatorValues',
      keyConditionExpression: 'symbol = :symbol',
      reverseOrder: true,
      limit: 1,
    };

    for (let c = 0; c < indicatorIds.length; c++) {
      // Prepare and insert row
      let indicatorId = indicatorIds[c];
      let indicatorDate;

      queryDetails.expressionAttributeValues = {
        ':symbol': indicatorId,
      };

      let result = await dynamodb.queryTable(queryDetails);

      if (result.length > 0) {
        indicatorDate = result[0]['valueDate'];
      } else {
        indicatorDate = defaultStartDate;
      }

      indicatorDate = utils.returnDateAsString(
        utils.dateAdd(indicatorDate, 'days', 1));

      indicatorDates.push({
        'symbol': indicatorId,
        'startDate': indicatorDate,
      });
    }

    return indicatorDates;
  } catch (err) {
    console.log(err);
  }
};


/**
 * Returns a list of financial indicators which were / are active for a
 * specific date
 * @param {String} valueDate the date to retrieve indicators for (yyyy-mm-dd)
 * @return {Object}  Object in form of:
 *    {
 *      indicatorId: indicatorValue,
 *      ...
 *    }
 */
let returnIndicatorValuesForDate = async function(valueDate) {
  if (!valueDate || !utils.isDate(valueDate)) {
    throw new Error('valueDate supplied is invalid: ' + valueDate);
  }

  try {
    let indicatorValues = {};
    let indicatorIds = await getIndicatorList();

    let queryDetails = {
      tableName: 'financialIndicatorValues',
      indexName: 'symbol-created-index',
      keyConditionExpression: 'symbol = :symbol and created <= :created',
      reverseOrder: true,
      limit: 1,
    };

    for (let c = 0; c < indicatorIds.length; c++) {
      // Prepare and insert row
      let indicatorId = indicatorIds[c];
      queryDetails.expressionAttributeValues = {
        ':symbol': indicatorId,
        ':created': valueDate,
      };

      let result = await dynamodb.queryTable(queryDetails);

      if (result.length > 0) {
        indicatorValues[indicatorId] = result[0]['value'];
      }
    }

    return indicatorValues;
  } catch (err) {
    console.log(err);
  }
};

/**
 * Retrieves and processes each financial indicator to look up new values
 */
let updateIndicatorValues = async function() {
  try {
    console.log('----- Start financial indicator retrieval -----');
    let indicatorIds = await getIndicatorList();
    let indicatorDates = await getIndicatorRequestDates(indicatorIds);

    for (let id = 0; id < indicatorDates.length; id++) {
      let indicatorDate = indicatorDates[id];
      let retrievalDetails = {};
      let symbol = indicatorDate.symbol;

      retrievalDetails.symbol = symbol;
      retrievalDetails.url = baseUrl + indicatorRetrievalValues[symbol].url +
        urlSuffix + dateParameter + indicatorDate['startDate'];
      retrievalDetails.dataColumn = indicatorRetrievalValues[symbol]
        .dataColumn;
      let indicatorResults = await
      retrieveQuandlIndicator(retrievalDetails);

      // Insert the value in the db
      for (let ic = 0; ic < indicatorResults.length; ic++) {
        let indicatorResult = indicatorResults[ic];
        await insertIndicatorValue(indicatorResult);
      };
    };
  } catch (err) {
    console.log(err);
  }
};

/**
 * Inserts an indicator value
 * @param {Object} indicatorValue value to insert in form of:
 *    {
 *      indicatorId: indicatorId,
 *      value_date: valueDate,
 *      value: value,
 *    }
 */
let insertIndicatorValue = async function(indicatorValue) {
  if (!indicatorValue['symbol'] || !indicatorValue['valueDate'] ||
    !indicatorValue['value']) {
    console.log('indicatorValue parameters missing: ' +
      JSON.stringify(indicatorValue));
    return;
  }

  console.log('----- Start write financial indicator results -----');

  try {
    indicatorValue['yearMonth'] = indicatorValue['valueDate']
      .substring(0, 7)
      .replace('-', '');

    let insertDetails = {
      tableName: 'financialIndicatorValues',
      values: indicatorValue,
      primaryKey: ['symbol', 'valueDate'],
    };

    await dynamodb.insertRecord(insertDetails);
  } catch (err) {
    console.log(err);
  }
};


module.exports = {
  updateIndicatorValues: updateIndicatorValues,
  returnIndicatorValuesForDate: returnIndicatorValuesForDate,
};
