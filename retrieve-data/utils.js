const json2csv = require('json2csv');
const fs = require('fs');
const symbols = require('./symbols.json');
const dbConn = require('./mysql-connection');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const moment = require('moment-timezone');
const credentials = require('../credentials/credentials.json');
const host = credentials.host;
const db = credentials.db;
const username = credentials.username;
const password = credentials.password;

let sleep = function(ms) {
  if (!ms) {
    ms = 1;
  }
  return new Promise((r) => {
    setTimeout(r, ms);
  });
};

let getLastRetrievalDate = function() {
  return new Promise(function(resolve, reject) {
    let connection;
    let retrievalDate;
    dbConn.connectToDb(host, username, password, db)
      .then((conn) => {
        connection = conn;

        return dbConn.selectQuery(connection, 'SELECT last_retrieval_date ' +
          'FROM `sharecast`.`last_retrieval_date` ' +
          'LIMIT 1;');
      })
      .then((rows) => {
        // console.log(rows);
        if (rows.length > 0) {
          retrievalDate = rows[0]['last_retrieval_date'];
        } else {
          retrievalDate = '';
        }
        dbConn.closeConnection(connection);
        resolve(retrievalDate);
      })
      .catch((err) => {
        console.log(err);
        dbConn.closeConnection(connection);
        reject(err);
      });
  });
};

let setLastRetrievalDate = asyncify(function(retrievalDate) {
  let connection;

  if (!retrievalDate) {
    console.log('Parameter retrievalDate not supplied');
    return;
  }
  try {
    connection = awaitify(dbConn.connectToDb(host, username, password, db));

    awaitify(dbConn.executeQuery(connection, 'DELETE  ' +
      'FROM `sharecast`.`last_retrieval_date`;'));

    awaitify(dbConn.executeQuery(connection, 'INSERT INTO ' +
      '`sharecast`.`last_retrieval_date`  VALUES(' +
      '\'' + retrievalDate + '\'); '));
  } catch (err) {
    console.log(err);
  } finally {
    if (connection) {
      dbConn.closeConnection(connection);
    }
  }
});

let getCompanies = function() {
  return new Promise(function(resolve, reject) {
    let connection;
    let companies = [];
    dbConn.connectToDb(host, username, password, db)
      .then((conn) => {
        connection = conn;

        return dbConn.selectQuery(connection, 'SELECT CompanySymbol, ' +
          'CompanySymbolYahoo FROM sharecast.companies;');
      })
      .then((rows) => {
        // console.log(rows);
        rows.forEach((row) => {
          companies.push({
            'symbol': row.CompanySymbol,
            'yahoo-symbol': row.CompanySymbolYahoo,
          });
        });
        dbConn.closeConnection(connection);
        resolve(companies);
      })
      .catch((err) => {
        console.log(err);
        dbConn.closeConnection(connection);
        reject(err);
      });
  });
};

let getIndices = function() {
  return new Promise(function(resolve, reject) {
    let connection;
    let indices = [];
    dbConn.connectToDb(host, username, password, db)
      .then((conn) => {
        connection = conn;

        return dbConn.selectQuery(connection, 'SELECT IndexSymbol, ' +
          'IndexSymbolYahoo FROM sharecast.indices;');
      })
      .then((rows) => {
        console.log(rows);
        rows.forEach((row) => {
          indices.push({
            'symbol': row.IndexSymbol,
            'yahoo-symbol': row.IndexSymbolYahoo,
          });
        });
        dbConn.closeConnection(connection);
        resolve(indices);
      })
      .catch((err) => {
        console.log(err);
        dbConn.closeConnection(connection);
        reject(err);
      });
  });
};

/** Returns a date in a set format: YYYY-MM-DD
* @param {String} dateValue - the date string to parse
* @param {String} dateFormat - a string defining the input format:
*  YYYY	2014	4 or 2 digit year
*  YY	14	2 digit year
*  Y	-25	Year with any number of digits and sign
*  Q	1..4	Quarter of year. Sets month to first month in quarter.
*  M MM	1..12	Month number
*  MMM MMMM	Jan..December	Month name in locale set by moment.locale()
*  D DD	1..31	Day of month
*  Do	1st..31st	Day of month with ordinal
*  DDD DDDD	1..365	Day of year
*  X	1410715640.579	Unix timestamp
*  x	1410715640579	Unix ms timestamp
* @return {String} the reformatted date or an empty string
*/
let returnDateAsString = function(dateValue, dateFormat) {
  if (moment(dateValue, dateFormat || '').isValid()) {
    return moment(dateValue, dateFormat || '').format('YYYY-MM-DD');
  } else {
    return '';
  }
};


/** Returns a date for the end of the month in a set format: YYYY-MM-DD
* @param {String} dateValue - the date string to parse
* @param {String} dateFormat - a string defining the input format:
*  YYYY	2014	4 or 2 digit year
*  YY	14	2 digit year
*  Y	-25	Year with any number of digits and sign
*  Q	1..4	Quarter of year. Sets month to first month in quarter.
*  M MM	1..12	Month number
*  MMM MMMM	Jan..December	Month name in locale set by moment.locale()
*  D DD	1..31	Day of month
*  Do	1st..31st	Day of month with ordinal
*  DDD DDDD	1..365	Day of year
*  X	1410715640.579	Unix timestamp
*  x	1410715640579	Unix ms timestamp
* @return {String} the formatted end of month date or an empty string
*/
let returnEndOfMonth = function(dateValue, dateFormat) {
  if (moment(dateValue, dateFormat || '').isValid()) {
    return moment(dateValue, dateFormat || '')
      .endOf('month')
      .format('YYYY-MM-DD');
  } else {
    return '';
  }
};

/**
 * converts the date string used for querying to a formated date string which
 *  can be displayed
 * @param {String} dateValue a date or string in a format which can be
 *   converted to a date
 * @param {string} unit the unit to change by "seconds", "minutes", "hours",
 "days" , "weeks", "months", "years"
 * @param {number} number to change, positive number for futures, negative
 *    number for past
 * @return {Date} a date with the new value
 */
let dateDiff = function(dateValue1, dateValue2, unit) {
  // Check that this really is a date
  if (!moment(dateValue1).isValid()) {
    throw new Error('dateValue1 invalid: ' + dateValue1);
  }

  if (!moment(dateValue2).isValid()) {
    throw new Error('dateValue2 invalid: ' + dateValue2);
  }

  if (!(unit === 'seconds' || unit === 'minutes' || unit === 'hours' ||
    unit === 'days' || unit === 'weeks' || unit === 'months' ||
    unit === 'years')) {
    throw new Error('unit invalid: ' + unit);
  }

  return Math.abs(dateValue1.diff(dateValue2, unit));
};


/**
 * converts the date string used for querying to a formated date string which
 *  can be displayed
 * @param {String} dateValue a date or string in a format which can be
 *   converted to a date
 * @param {string} unit the unit to change by "days" days, "weeks" weeks,
 *                    "months", "year" years
 * @param {number} number to change, positive number for futures, negative
 *    number for past
 * @return {Date} a date with the new value
 */
let dateAdd = function(dateValue, unit, number) {
  // Check that this really is a date
  if (!moment(dateValue).isValid()) {
    throw new Error('dateValue invalid: ' + dateValue);
  }
  if (!(unit === 'days' || unit === 'weeks' || unit === 'months' ||
    unit === 'years')) {
    throw new Error('unit invalid: ' + unit);
  }
  if (typeof number !== 'number') {
    throw new Error('number invalid: ' + number);
  }

  return moment(dateValue).add(number, unit).format('YYYY-MM-DD');
};

let isDate = function(dateValue, dateFormat) {
  // Check that this really is a date
  return moment(dateValue, dateFormat || '').isValid();
};

let createFieldArray = function(fieldObject) {
  return Object.keys(fieldObject);
};

let writeToCsv = function(csvData, csvFields, filePrefix, dateString) {
  // If no date string supplied, use today's date
  let today = new Date();

  dateString = dateString || today.getFullYear() + '-' +
    ('0' + (today.getMonth() + 1)).slice(-2) + '-' +
    ('0' + today.getDate()).slice(-2);

  let csvOutput = json2csv({
    data: csvData,
    fields: csvFields,
  });

  fs.writeFile('../data/' + filePrefix + '-' + dateString + '.csv',
    csvOutput, function(err) {
      if (err)
        throw err;
      console.log('File saved');
    });
};

let writeJSONfile = function(jsonObject, fileName) {
  fs.writeFile(fileName, JSON.stringify(jsonObject), function(err) {
    if (err)
      throw err;
    console.log('JSON File saved');
  });
};

let doesDataFileExist = function(fileName) {
  return fs.existsSync('../data/' + fileName);
};

let checkForNumber = function(value) {
  let finalChar = String(value).slice(-1);
  let leadingVal = String(value).slice(0, String(value).length - 1);
  let adjustedVal = value;

  let charsToMatch = {
    'k': 1000,
    'K': 1000,
    'm': 1000000,
    'M': 1000000,
    'b': 1000000000,
    'B': 1000000000,
  };
  let possibleChars = Object.keys(charsToMatch);

  // Check if final character is thousands, millions or billions
  if (!isNaN(leadingVal) && possibleChars.indexOf(finalChar) > -1) {
    // if it is, multiple value to get normal number
    adjustedVal = leadingVal * charsToMatch[finalChar];
  }

  // If it's not a number, replace commas in value, then check if it's a number
  if (isNaN(adjustedVal)) {
    let holdingVal = adjustedVal.replace(/,/g, '');

    if (!isNaN(holdingVal)) {
      adjustedVal = holdingVal;
    }
  }
  return adjustedVal;
};


module.exports = {
  sleep: sleep,
  getLastRetrievalDate: getLastRetrievalDate,
  setLastRetrievalDate: setLastRetrievalDate,
  getCompanies: getCompanies,
  getIndices: getIndices,
  returnDateAsString: returnDateAsString,
  dateAdd: dateAdd,
  dateDiff: dateDiff,
  isDate: isDate,
  createFieldArray: createFieldArray,
  writeToCsv: writeToCsv,
  writeJSONfile: writeJSONfile,
  doesDataFileExist: doesDataFileExist,
  checkForNumber: checkForNumber,
  returnEndOfMonth: returnEndOfMonth,
};
