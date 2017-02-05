const csv = require('csvtojson');
const utils = require('./utils');
const dbConn = require('./mysql-connection');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const credentials = require('../credentials/credentials.json');
const host = credentials.host;
const db = credentials.db;
const username = credentials.username;
const password = credentials.password;


/** Example SQL insert for index quotes table **
INSERT INTO `sharecast`.`index_quotes`
(`index_symbol`,
`quote_date`,
`year_month`,
`previousClose`,
`change`,
`changeInPercent`,
`daysLow`,
`daysHigh`,
`52WeekHigh`,
`52WeekLow`,
`changeFrom52WeekLow`,
`changeFrom52WeekHigh`,
`percentChangeFrom52WeekLow`,
`percentChangeFrom52WeekHigh`)
VALUES
();

Values present in index history
--------------------------
INSERT INTO `sharecast`.`index_quotes`
(`index_symbol`,
`quote_date`,
`year_month`,
`previousClose`,
`daysLow`,
`daysHigh`,
)
VALUES
(symbol,
date,
convert date to yearmonth,
open,
low,
high
);
**/

let retrieveCsv = function(filePath) {
  return new Promise(function(resolve, reject) {
    let csvJsonData = [];
    csv()
      .fromFile(filePath)
      .on('json', (jsonObj) => {
        // console.log(jsonObj);
        csvJsonData.push(jsonObj);
      })
      .on('done', (error) => {
        if (error) {
          reject(error);
        } else {
          resolve(csvJsonData);
        }
      });
  });
};

let readAndInsertIndexHistory = asyncify(function() {
  let connection;
  try {
    let csvFilePath = '../data/indice-history-2017-01-31.csv';
    let csvData = awaitify(retrieveCsv(csvFilePath));
    // console.log(csvData);

    // Open DB connection
    connection = awaitify(dbConn.connectToDb(host, username, password, db));

    for (let c = 0; c < csvData.length; c++) {
      // Prepare and insert row
      let csvRow = csvData[c];
      let quoteDate = utils.returnDateAsString(csvRow['date']);
      let yearMonth = quoteDate.substring(0, 7).replace('-', '');
      awaitify(dbConn.executeQuery(connection,
        'INSERT INTO `sharecast`.`index_quotes`' +
        '(`index_symbol`,' +
        '`quote_date`,' +
        '`year_month`,' +
        '`previousClose`,' +
        '`daysLow`,' +
        '`daysHigh`' +
        ')' +
        '  VALUES' +
        '(\'' + csvRow['symbol'] + '\',' +
        '\'' + quoteDate + '\',' +
        '\'' + yearMonth + '\',' +
        '' + csvRow['open'] + ',' +
        '' + csvRow['low'] + ',' +
        '' + csvRow['high'] +
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

let testMe = asyncify(function() {
  let connection;
  let rows;
  try {
    connection = awaitify(dbConn.connectToDb(host, username, password, db));
    rows = awaitify(dbConn.selectQuery(connection, 'SELECT last_retrieval_date '
      + 'FROM `sharecast`.`last_retrieval_date` LIMIT 1;'));
    console.log(rows);
    awaitify(dbConn.executeQuery(connection, 'UPDATE ' +
      '`sharecast`.`last_retrieval_date` SET last_retrieval_date = ' +
      '\'2017-02-02\' WHERE last_retrieval_date = \'2017-02-01\';'));
  } catch (err) {
    console.log(err);
  } finally {
    if (connection) {
      dbConn.closeConnection(connection);
    }
  }
});


readAndInsertIndexHistory();
