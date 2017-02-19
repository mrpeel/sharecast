const csv = require('csvtojson');
const utils = require('./utils');
const dynamodb = require('./dynamodb');
const dynamoMetrics = require('./dynamo-retrieve-google-company-data');
// const metrics = require('./retrieve-google-company-data');
const companyHistory = require('../data/company-history.json');
const historyTranslation = require('./history-translation.json');
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

let readAndInsertDividendHistory = asyncify(function() {
  let connection;
  for (companyCounter = 0; companyCounter <= 3030; companyCounter += 10) {
    let csvFileName = 'companies-dividend-history-' + companyCounter +
      '-2017-02-03.csv';

    if (utils.doesDataFileExist(csvFileName)) {
      try {
        let csvFilePath = '../data/' + csvFileName;
        let csvData = awaitify(retrieveCsv(csvFilePath));
        // console.log(csvData);

        // Open DB connection
        connection = awaitify(dbConn.connectToDb(host, username, password, db));

        for (let c = 0; c < csvData.length; c++) {
          // Prepare and insert row
          let csvRow = csvData[c];
          let dividendDate = utils.returnDateAsString(csvRow['date']);

          awaitify(dbConn.executeQuery(connection,
            'INSERT INTO `sharecast`.`dividend_history` ' +
            '(`company_symbol`, ' +
            '`dividend_date`, ' +
            '`value`) ' +
            '  VALUES' +
            '(\'' + csvRow['symbol'] + '\',' +
            '\'' + dividendDate + '\',' +
            +csvRow['dividends'] +
            ');'));
        }
      } catch (err) {
        console.log(err);
      } finally {
        if (connection) {
          dbConn.closeConnection(connection);
        }
      }
    } else {
      console.log(csvFileName + ' does not exist');
    }
  }
});

/* let readAndInsertMetrics = asyncify(function() {
  let csvFileName = 'company-metrics-2017-01-26.csv';
  let csvFileName2 = 'company-metrics-2017-01-25.csv';
  let companyEps = {};

  if (utils.doesDataFileExist(csvFileName) &&
    utils.doesDataFileExist(csvFileName2)) {
    try {
      let csvFilePath = '../data/' + csvFileName;
      let metricsValues = awaitify(retrieveCsv(csvFilePath));

      metricsValues.forEach((metricValue) => {
        metricValue['metricsDate'] = '2017-01-26';
        metricValue['yearMonth'] = metricValue['metricsDate']
          .substring(0, 7)
          .replace('-', '');

        companyEps[metricValue['symbol']] = metricValue['EPS'];

        // Remove nulls, empty values and -
        // Check through for values with null and remove from object
        Object.keys(metricValue).forEach((field) => {
          let holdingVal = metricValue[field];
          if (holdingVal === null || holdingVal === '' ||
            holdingVal === '-') {
            delete metricValue[field];
          } else if (typeof (holdingVal) === 'string' &&
            !isNaN(holdingVal.replace(',', ''))) {
            metricValue[field] = Number(holdingVal.replace(',', ''));
          }
        });

        // console.log(metricValue);

        awaitify(dynamoMetrics.insertCompanyMetricsValue(metricValue));
      });

      csvFilePath = '../data/' + csvFileName2;
      metricsValues = awaitify(retrieveCsv(csvFilePath));

      metricsValues.forEach((metricValue) => {
        metricValue['metricsDate'] = '2017-01-25';
        metricValue['yearMonth'] = metricValue['metricsDate']
          .substring(0, 7)
          .replace('-', '');

        metricValue['EPS'] = companyEps[metricValue['symbol']];

        // Remove nulls, empty values and -
        // Check through for values with null and remove from object
        Object.keys(metricValue).forEach((field) => {
          let holdingVal = metricValue[field];
          if (holdingVal === null || holdingVal === '' ||
            holdingVal === '-') {
            delete metricValue[field];
          } else if (typeof (holdingVal) === 'string' &&
            !isNaN(holdingVal.replace(',', ''))) {
            metricValue[field] = Number(holdingVal.replace(',', ''));
          }
        });

        // console.log(metricValue);

        awaitify(dynamoMetrics.insertCompanyMetricsValue(metricValue));
      });
    } catch (err) {
      console.log(err);
    }
  }
}); */

let extractAndInsertMetrics = asyncify(function() {
  try {
    let metricsValues = awaitify(metrics.returnAllCompanyMetricsValues());

    metricsValues.forEach((metricValue) => {
      metricValue['symbol'] = metricValue['CompanySymbol'];
      metricValue['metricsDate'] = utils.returnDateAsString(
        metricValue['MetricsDate']);
      metricValue['yearMonth'] = metricValue['metricsDate']
        .substring(0, 7)
        .replace('-', '');
      metricValue['created'] = utils.returnDateAsString(
        metricValue['Created']);

      // Remove uppercase properties
      delete metricValue['Id'];
      delete metricValue['CompanySymbol'];
      delete metricValue['MetricsDate'];
      delete metricValue['Created'];

      // Remove nulls, empty values and -
      // Check through for values with null and remove from object
      Object.keys(metricValue).forEach((field) => {
        let holdingVal = metricValue[field];
        if (holdingVal === null || holdingVal === '' ||
          holdingVal === '-') {
          delete metricValue[field];
        } else if (typeof (holdingVal) === 'string' &&
          !isNaN(holdingVal.replace(',', ''))) {
          metricValue[field] = Number(holdingVal.replace(',', ''));
        }
      });

      // console.log(metricValue);

      awaitify(dynamoMetrics.insertCompanyMetricsValue(metricValue));
    });
  } catch (err) {
    console.log(err);
  }
});


let extractAndInsertIndexHistory = asyncify(function() {
  let connection;
  let insertDetails = {
    tableName: 'indexQuotes',
    values: {},
    primaryKey: [
      'symbol', 'quoteDate',
    ],
  };

  try {
    // Open DB connection
    connection = awaitify(dbConn.connectToDb(host, username, password, db));

    let results = awaitify(dbConn.selectQuery(connection,
      'SELECT * FROM `sharecast`.`index_quotes`;'));

    if (results.length) {
      results.forEach((indexValue) => {
        indexValue['symbol'] = indexValue['IndexSymbol'];
        indexValue['quoteDate'] = utils.returnDateAsString(
          indexValue['QuoteDate']);
        indexValue['yearMonth'] = indexValue['quoteDate']
          .substring(0, 7)
          .replace('-', '');
        indexValue['created'] = utils.returnDateAsString(
          indexValue['Created']);
        indexValue['percebtChangeFrom52WeekHigh'] = indexValue['percentChangeFrom52WeekHigh'];

        // Remove uppercase properties
        delete indexValue['Id'];
        delete indexValue['IndexSymbol'];
        delete indexValue['QuoteDate'];
        delete indexValue['YearMonth'];
        delete indexValue['Created'];
        delete indexValue['percentChangeFrom52WeekHigh'];

        // Remove nulls, empty values and -
        // Check through for values with null and remove from object
        Object.keys(indexValue).forEach((field) => {
          let holdingVal = indexValue[field];
          if (holdingVal === null || holdingVal === '' ||
            holdingVal === '-') {
            delete indexValue[field];
          } else if (typeof (holdingVal) === 'string' &&
            !isNaN(holdingVal.replace(',', ''))) {
            indexValue[field] = Number(holdingVal.replace(',', ''));
          }
        });

        // console.log(indexValue);
        insertDetails.values = indexValue;

        awaitify(dynamodb.insertRecord(insertDetails));
      });
    }
  } catch (err) {
    console.log(err);
  } finally {
    if (connection) {
      dbConn.closeConnection(connection);
    }
  }
});


let openAndInsertIndexHistory = asyncify(function() {
  let insertDetails = {
    tableName: 'indexQuotes',
    values: {},
    primaryKey: [
      'symbol', 'quoteDate',
    ],
  };

  let csvFileNames = ['indices-2017-02-03.csv'];

  csvFileNames.forEach((csvFileName) => {
    if (utils.doesDataFileExist(csvFileName)) {
      try {
        let csvFilePath = '../data/' + csvFileName;
        let indexValues = awaitify(retrieveCsv(csvFilePath));

        indexValues.forEach((indexValue) => {
          indexValue['quoteDate'] = utils.returnDateAsString(
            indexValue['lastTradeDate']);
          indexValue['yearMonth'] = indexValue['quoteDate']
            .substring(0, 7)
            .replace('-', '');

          // Remove nulls, empty values and -
          // Check through for values with null and remove from object
          Object.keys(indexValue).forEach((field) => {
            let holdingVal = indexValue[field];
            if (holdingVal === null || holdingVal === '' ||
              holdingVal === '-') {
              delete indexValue[field];
            } else if (typeof (holdingVal) === 'string' &&
              !isNaN(holdingVal.replace(',', ''))) {
              indexValue[field] = Number(holdingVal.replace(',', ''));
            }
          });

          // console.log(indexValue);
          insertDetails.values = indexValue;

          awaitify(dynamodb.insertRecord(insertDetails));
        });
      } catch (err) {
        console.log(err);
      }
    }
  });
});

let extractAndInsertCompanyHistory = asyncify(function() {
  let queryDetails = {
    tableName: 'companyMetrics',
    limit: 1,
  };

  let updateDetails = {
    tableName: 'companyMetrics',
  };

  let updateCounter = 0;
  let insertCounter = 0;

  try {
    Object.keys(companyHistory).forEach((company) => {
      let companyMetrics = companyHistory[company];

      Object.keys(companyMetrics).forEach((yearMetric) => {
        let metricValue = companyMetrics[yearMetric];

        // Translate values into metrics property names
        Object.keys(metricValue).forEach((field) => {
          // Check if a translation is found
          if (historyTranslation[field] && metricValue[field] !== '--' &&
            metricValue[field] !== '-') {
            let val = utils.checkForNumber(metricValue[field]);
            val = Number(val) * historyTranslation[field]['multiply-by'];
            metricValue[historyTranslation[field]['name']] = val;
          }
          // emove original field from object
          delete metricValue[field];
        });

        // Remove nulls, empty values and -
        // Check through for values with null and remove from object
        Object.keys(metricValue).forEach((field) => {
          let holdingVal = metricValue[field];
          if (holdingVal === null || holdingVal === '' ||
            holdingVal === '-' || holdingVal === '--') {
            delete metricValue[field];
          } else if (typeof (holdingVal) === 'string' &&
            !isNaN(holdingVal.replace(',', ''))) {
            metricValue[field] = Number(holdingVal.replace(',', ''));
          }
        });

        /* Dates are stored in the format 06/07 = June 2007 */
        let baseDate = utils.returnDateAsString('01/' + yearMetric,
          'D/M/YYYY');
        let metricsEndDate = utils.dateAdd(baseDate, 'months', 3);
        let metricsStartDate = utils.dateAdd(baseDate, 'months', -1);


        // Try to find a matching dividend record in the companyMetrics table
        // Look forward up to 30 days
        queryDetails.keyConditionExpression = 'symbol = :symbol and ' +
          'metricsDate >= :metricsDate';
        queryDetails.expressionAttributeValues = {
          ':symbol': company,
          ':metricsDate': baseDate,
        };

        let result = awaitify(dynamodb.queryTable(queryDetails));

        // Check if we have a result within the time period
        if (result.length === 0 || result[0]['metricsDate'] > metricsEndDate) {
          // Not found, so look back up to 30 days
          queryDetails.keyConditionExpression = 'symbol = :symbol and ' +
            'metricsDate < :metricsDate';
          queryDetails.expressionAttributeValues = {
            ':symbol': company,
            ':metricsDate': baseDate,
          };
          queryDetails.reverseOrder = true;

          result = awaitify(dynamodb.queryTable(queryDetails));
        }

        // Check if we have a result within the time period
        if (result.length > 0 && (result[0]['metricsDate'] < metricsEndDate ||
          result[0]['metricsDate'] > metricsStartDate)) {
          // We have a result so we'll update that record
          let returnedMetricDate = result[0]['metricsDate'];
          // Set-up the values for recent quarter dividend
          let dividendRecentQuarter = result[0]['DividendPerShare'] || 0;
          // Set up the key: symbol and metricsDate
          updateDetails.key = {
            symbol: company,
            metricsDate: returnedMetricDate,
          };
          let fieldsPresent = ['#DividendRecentQuarter=:DividendRecentQuarter'];
          let updateExpression;
          let expressionAttributeValues = {
            ':DividendRecentQuarter': dividendRecentQuarter,
          };
          let expressionAttributeNames = {
            '#DividendRecentQuarter': 'DividendRecentQuarter',
          };

          // Get a list of fields
          Object.keys(metricValue).forEach((field) => {
            expressionAttributeValues[(':' + field)] = metricValue[field];
            expressionAttributeNames[('#' + field)] = field;
            fieldsPresent.push('#' + field + '=:' + field);
          });

          // Enure that some fields are present to update
          if (fieldsPresent.length) {
            updateExpression = 'set ' + fieldsPresent.join(',');

            updateDetails.updateExpression = updateExpression;
            updateDetails.expressionAttributeValues = expressionAttributeValues;
            updateDetails.expressionAttributeNames = expressionAttributeNames;

            updateCounter++;
            awaitify(dynamodb.updateRecord(updateDetails));
          }
        } else {
          // Check that the object has some values to insert
          if (Object.keys(metricValue).length) {
            /* No result so estimate the date.
            Dates are stored in the format 06/07 = June 2007,
              Average it out and set to the middle of the month */
            let estDate = utils.returnDateAsString('15/' + yearMetric,
              'D/M/YYYY');
            metricValue['metricsDate'] = estDate;
            metricValue['yearMonth'] = estDate
              .substring(0, 7)
              .replace('-', '');

            // Add in all the required base values
            metricValue['symbol'] = company;

            insertCounter++;
            awaitify(dynamoMetrics.insertCompanyMetricsValue(metricValue));
          }
        }

      // console.log(metricValue);
      /* Make sure can't exceed 5 writes per second - put latency is around 15
         milliseconds  */
      // awaitify(utils.sleep(175));
      });
    });

    console.log('Updates: ', updateCounter);
    console.log('Inserts: ', insertCounter);
  } catch (err) {
    console.log(err);
  }
});


// readAndInsertIndexHistory();

// readAndInsertDividendHistory();

// extractAndInsertMetrics();

// extractAndInsertIndexHistory();

// openAndInsertIndexHistory();

extractAndInsertCompanyHistory();
