const json2csv = require('json2csv');
const fs = require('fs');
const symbols = require('./symbols.json');
const dbConn = require('./mysql-connection');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const credentials = require('../credentials/credentials.json');
const host = credentials.host;
const db = credentials.db;
const username = credentials.username;
const password = credentials.password;

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

        return dbConn.selectQuery(connection, 'SELECT company_symbol, ' +
          'company_symbol_yahoo FROM sharecast.companies;');
      })
      .then((rows) => {
        // console.log(rows);
        rows.forEach((row) => {
          companies.push({
            'symbol': row.company_symbol,
            'yahoo-symbol': row.company_symbol_yahoo,
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

        return dbConn.selectQuery(connection, 'SELECT index_symbol, ' +
          'index_symbol_yahoo FROM sharecast.indices;');
      })
      .then((rows) => {
        console.log(rows);
        rows.forEach((row) => {
          indices.push({
            'symbol': row.index_symbol,
            'yahoo-symbol': row.index_symbol_yahoo,
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

let writeIndexResults = asyncify(function(indexData) {
  let connection;
  try {
    // Open DB connection
    connection = awaitify(dbConn.connectToDb(host, username, password, db));

    for (let c = 0; c < csvData.length; c++) {
      // Prepare and insert row
      let csvRow = csvData[c];
      let quoteDate = utils.returnDateAsString(csvRow['lastTradeDate']);
      let yearMonth = quoteDate.substring(0, 7).replace('-', '');
      awaitify(dbConn.executeQuery(connection,
        'INSERT INTO `sharecast`.`index_quotes`' +
        '(`index_symbol`,' +
        '`quote_date`,' +
        '`year_month`,' +
        '`previousClose`,' +
        '`change`,' +
        '`changeInPercent`,' +
        '`daysLow`,' +
        '`daysHigh`,' +
        '`52WeekHigh`,' +
        '`52WeekLow`,' +
        '`changeFrom52WeekLow`,' +
        '`changeFrom52WeekHigh`,' +
        '`percentChangeFrom52WeekLow`,' +
        '`percentChangeFrom52WeekHigh`' +
        ')' +
        '  VALUES' +
        '(\'' + csvRow['symbol'] + '\',' +
        '\'' + quoteDate + '\',' +
        '\'' + yearMonth + '\',' +
        '' + csvRow['previousClose'] + ',' +
        '' + csvRow['change'] + ',' +
        '' + csvRow['changeInPercent'] + ',' +
        '' + csvRow['daysLow'] + ',' +
        '' + csvRow['daysHigh'] + ',' +
        '' + csvRow['52WeekHigh'] + ',' +
        '' + csvRow['52WeekLow'] + ',' +
        '' + csvRow['changeFrom52WeekLow'] + ',' +
        '' + csvRow['changeFrom52WeekHigh'] + ',' +
        '' + csvRow['percentChangeFrom52WeekLow'] + ',' +
        '' + csvRow['percebtChangeFrom52WeekHigh'] +
        ');'));
    }
  } catch (err) {
    console.log(err);
  } finally {
    if (connection) {
      dbConn.closeConnection(connection);
    }
  }
});

let returnDateAsString = function(dateValue) {
  let checkDate = new Date(dateValue);

  return checkDate.getFullYear() + '-' +
    ('0' + (checkDate.getMonth() + 1)).slice(-2) + '-' +
    ('0' + checkDate.getDate()).slice(-2);
};

/**
 * converts the date string used for querying to a formated date string which
 *  can be displayed
 * @param {String} dateValue a date or string in a format which can be
 *   converted to a date
 * @param {string} unit the unit to change by "d" days, "w" weeks, "y" years
 * @param {number} number to change, positive number for futures, negative
 *    number for past
 * @return {Date} a date with the new value
 */
let dateAdd = function(dateValue, unit, number) {
  // Check that this really is a date
  if (!isDate(dateValue)) {
    throw new Error('dateValue invalid: ' + dateValue);
  }
  if (!(unit === 'd' || unit === 'm' || unit === 'y')) {
    throw new Error('unit invalid: ' + unit);
  }
  if (typeof number !== 'number') {
    throw new Error('number invalid: ' + number);
  }

  let newDate = new Date(dateValue);
  let dateComponents = {};

  dateComponents.years = newDate.getFullYear();
  dateComponents.months = newDate.getMonth();
  dateComponents.days = newDate.getDate();

  if (unit === 'd') {
    newDate.setDate(dateComponents.days + number);
  } else if (unit === 'm') {
    newDate.setMonth(dateComponents.months + number);
  } else if (unit === 'y') {
    newDate.setFullYear(dateComponents.years + number);
  }

  return newDate;
};

let isDate = function(dateValue) {
  let newDate = new Date(dateValue);

  // Check that this really is a date - it must be able to return the month
  if (isNaN(newDate.getMonth())) {
    return false;
  } else {
    return true;
  }
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
    return leadingVal * charsToMatch[finalChar];
  } else {
    return value;
  }
};


module.exports = {
  getLastRetrievalDate: getLastRetrievalDate,
  setLastRetrievalDate: setLastRetrievalDate,
  getCompanies: getCompanies,
  getIndices: getIndices,
  writeIndexResults: writeIndexResults,
  returnDateAsString: returnDateAsString,
  dateAdd: dateAdd,
  isDate: isDate,
  createFieldArray: createFieldArray,
  writeToCsv: writeToCsv,
  writeJSONfile: writeJSONfile,
  doesDataFileExist: doesDataFileExist,
  checkForNumber: checkForNumber,
};
