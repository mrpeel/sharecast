const json2csv = require('json2csv');
const fs = require('fs');
const symbols = require('./symbols.json');
const dbConn = require('./mysql-connection');
const credentials = require('../credentials/credentials.json');
const host = 'localhost';
const db = 'sharecast';
const username = credentials.username;
const password = credentials.password;

module.exports = {
  getLastRetrievalDate: function() {
    return new Promise(function(resolve, reject) {
      let connection;
      let retrievalDate;
      dbConn.connectToDb(host, username, password, db)
        .then((conn) => {
          connection = conn;

          return dbConn.queryDb(connection, 'SELECT last_retrieval_date ' +
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
  },
  setLastRetrievalDate(retrievalDate) {
    let connection;

    if (!retrievalDate) {
      console.log('Parameter retrievalDate not supplied');
      return;
    }

    dbConn.connectToDb(host, username, password, db)
      .then((conn) => {
        connection = conn;

        return dbConn.queryDb(connection, 'DELETE  ' +
          'FROM `sharecast`.`last_retrieval_date`;');
      })
      .then((result) => {
        // console.log(result);
        return dbConn.queryDb(connection, 'INSERT INTO ' +
          '`sharecast`.`last_retrieval_date`  VALUES(' +
          '\'' + retrievalDate + '\'); ');
      })
      .then((result) => {
        // console.log(result);
        dbConn.closeConnection(connection);
      })
      .catch((err) => {
        console.log(err);
        dbConn.closeConnection(connection);
      });
  },
  getCompanies: function() {
    return symbols.companies;
  },
  getIndices: function() {
    return symbols.indices;
  },
  returnDateAsString: function(dateValue) {
    let checkDate = new Date(dateValue);

    return checkDate.getFullYear() + '-' +
      ('0' + (checkDate.getMonth() + 1)).slice(-2) + '-' +
      ('0' + checkDate.getDate()).slice(-2);
  },
  createFieldArray: function(fieldObject) {
    return Object.keys(fieldObject);
  },
  writeToCsv: function(csvData, csvFields, filePrefix, dateString) {
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
  },
  writeJSONfile: function(jsonObject, fileName) {
    fs.writeFile(fileName, JSON.stringify(jsonObject), function(err) {
      if (err)
        throw err;
      console.log('JSON File saved');
    });
  },
  doesDataFileExist: function(fileName) {
    return fs.existsSync('../data/' + fileName);
  },
  checkForNumber: function(value) {
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
  },
};
