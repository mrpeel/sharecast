
const fetch = require('node-fetch');
const dbConn = require('./mysql-connection');
const utils = require('./utils');
const credentials = require('../credentials/credentials.json');
const apiKey = credentials['quandl-api-key'];
const host = credentials.host;
const db = credentials.db;
const username = credentials.username;
const password = credentials.password;
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');

let baseUrl = 'https://www.quandl.com/api/v3/datasets/';
let urlSuffix = '?api_key=' + apiKey;
let defaultStartDate = '2011-09-29';
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
    monthLag: 1,
    dayLag: 1,
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
    url: 'RBA/H01_GGDPCVGDP.json?',
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
let retrieveQuandlIndicator = asyncfunction(indicatorDetails) {
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
            'indicatorId': indicatorDetails.indicatorId,
            'value-date': row[0],
            'value': row[indicatorDetails.dataColumn],
          });
        });

        resolve(indicatorData);
      })
      .catch((err) => {
        reject(err);
      });
  });
});

/**
 * Returns a list of financial indicators
 * @return {Array}  list of indicator codes to retrieve
 */
let getIndicatorList = asyncfunction() {
  return Object.keys(indicatorRetrievalValues);
});

/**
 * Returns a list of financial indicators
 * @param {Array} indicatorIds list of indicator codes to retrieve
 * @return {Object}  Object in form of:
 *    {
 *      indicator: indicatorId,
 *      start-date: startDate,
 *    }
 */
let getIndicatorRequestDates = asyncfunction(indicatorIds) {
  if (!indicatorIds) {
    throw new Error('No indicators supplied: ');
  }

  let connection;
  try {
    let indicatorDates = [];

    // Open DB connection
    connection = awaitdbConn.connectToDb(host, username, password, db));

    for (let c = 0; c < indicatorIds.length; c++) {
      // Prepare and insert row
      let indicatorId = indicatorIds[c];
      let indicatorDate;
      let result = awaitdbConn.selectQuery(connection,
        'SELECT `ValueDate` ' +
        'FROM `sharecast`.`financial_indicator_values` ' +
        'WHERE `IndicatorSymbol` = \'' + indicatorId + '\' ' +
        'ORDER BY `ValueDate` desc ' +
        'LIMIT 1;'
      ));
      if (result.length > 0) {
        indicatorDate = result[0]['ValueDate'];
      } else {
        indicatorDate = defaultStartDate;
      }

      indicatorDate = utils.returnDateAsString(
        utils.dateAdd(indicatorDate, 'days', 1));

      indicatorDates.push({
        'indicatorId': indicatorId,
        'start-date': indicatorDate,
      });
    }

    return indicatorDates;
  } catch (err) {
    console.log(err);
  } finally {
    if (connection) {
      dbConn.closeConnection(connection);
    }
  }
});


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
let returnIndicatorValuesForDate = asyncfunction(valueDate) {
  if (!valueDate || !utils.isDate(valueDate)) {
    throw new Error('valueDate supplied is invalid: ' + valueDate);
  }

  let connection;
  try {
    let indicatorValues = {};
    let indicatorIds = awaitgetIndicatorList());

    // Open DB connection
    connection = awaitdbConn.connectToDb(host, username, password, db));

    for (let c = 0; c < indicatorIds.length; c++) {
      // Prepare and insert row
      let indicatorId = indicatorIds[c];
      let result = awaitdbConn.selectQuery(connection,
        'SELECT `Value` ' +
        'FROM `sharecast`.`financial_indicator_values` ' +
        'WHERE `IndicatorSymbol` = \'' + indicatorId + '\' ' +
        'AND Created <= \'' + valueDate + '\'' +
        'ORDER BY `ValueDate` desc ' +
        'LIMIT 1;'
      ));
      if (result.length > 0) {
        indicatorValues[indicatorId] = result[0]['Value'];
      }
    }

    return indicatorValues;
  } catch (err) {
    console.log(err);
  } finally {
    if (connection) {
      dbConn.closeConnection(connection);
    }
  }
});

/**
 * Retrieves and processes each financial indicator to look up new values
 */
let updateIndicatorValues = asyncfunction() {
  try {
    let indicatorIds = awaitgetIndicatorList());
    let indicatorDates = awaitgetIndicatorRequestDates(indicatorIds));

    indicatorDates.forEach((indicatorDate) => {
      let retrievalDetails = {};
      let indicatorId = indicatorDate.indicatorId;

      retrievalDetails.indicatorId = indicatorId;
      retrievalDetails.url = baseUrl + indicatorRetrievalValues[indicatorId].url
      + urlSuffix + dateParameter + indicatorDate['start-date'];
      retrievalDetails.dataColumn = indicatorRetrievalValues[indicatorId]
        .dataColumn;
      let indicatorResults = await
        retrieveQuandlIndicator(retrievalDetails));

      // Insert the value in the db
      indicatorResults.forEach((indicatorResult) => {
        awaitinsertIndicatorValue(indicatorResult));
      });
    });
  } catch (err) {
    console.log(err);
  }
});

/**
 * Inserts an indicator value
 * @param {Object} indicatorValue value to insert in form of:
 *    {
 *      indicatorId: indicatorId,
 *      value_date: valueDate,
 *      value: value,
 *    }
 */
let insertIndicatorValue = asyncfunction(indicatorValue) {
  let connection;

  if (!indicatorValue['indicatorId'] || !indicatorValue['value-date'] ||
    !indicatorValue['value']) {
    console.log('indicatorValue parameters missing: ' +
      JSON.stringify(indicatorValue));
    return;
  }

  try {
    connection = awaitdbConn.connectToDb(host, username, password, db));

    indicatorValue['year-month'] = indicatorValue['value-date']
      .substring(0, 7)
      .replace('-', '');

    indicatorValue['Id'] = 'Fi' + indicatorValue['indicatorId'] +
    indicatorValue['value-date'].replace('-', '');


    awaitdbConn.executeQuery(connection, 'INSERT INTO ' +
      '`sharecast`.`financial_indicator_values` ' +
      '(`Id`, ' +
      '`IndicatorSymbol`, ' +
      '`ValueDate`, ' +
      '`YearMonth`, ' +
      '`Value`)  ' +
      'VALUES ( ' +
      '\'' + indicatorValue['Id'] + '\', ' +
      '\'' + indicatorValue['indicatorId'] + '\', ' +
      '\'' + indicatorValue['value-date'] + '\', ' +
      '\'' + indicatorValue['year-month'] + '\', ' +
      indicatorValue['value'] +
      ');'));
  } catch (err) {
    console.log(err);
  } finally {
    if (connection) {
      dbConn.closeConnection(connection);
    }
  }
});

module.exports = {
  updateIndicatorValues: updateIndicatorValues,
  returnIndicatorValuesForDate: returnIndicatorValuesForDate,
};
