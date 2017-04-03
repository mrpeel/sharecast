'use strict';

const aws = require('aws-sdk');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const json2csv = require('json2csv');

aws.config.update({
  region: 'ap-southeast-2',
});

let writeCsv = function(csvData, csvFields, bucket, exportName) {
  let today = new Date();
  let dateString = today.getFullYear() + '-' +
  ('0' + (today.getMonth() + 1)).slice(-2) + '-' +
  ('0' + today.getDate()).slice(-2);

  let csvOutput = json2csv({
    data: csvData,
    fields: csvFields,
  });

  let s3obj = new aws.S3({
    params: {
      Bucket: bucket,
      Key: exportName + '-' + dateString,
    },
  });

  s3obj.upload({
    Body: csvOutput,
  }, function(err) {
    if (err) {
      console.error(err);
    } else {
      console.log('All done.');
    }
  }).on('httpUploadProgress', function(evt) {
    console.log('Uploading...');
  });
};

aws.config.loadFromPath('../credentials/aws.json');

let dynamo = new aws.DynamoDB();
let dynamoDb = new aws.DynamoDB.DocumentClient();

let exportHandler = asyncify(function() {
  try {
    let tableName = 'companyQuotes';
    let csvData = [];
    let csvFields = [];
    let numRecs = 0;

    // create parameters hash for table scan
    let params = {
      TableName: tableName,
      ReturnConsumedCapacity: 'NONE',
      Limit: '1',
      FilterExpression: 'quoteDate BETWEEN :startDate AND :endDate',
      ExpressionAttributeValues: {
        ':startDate': '2007-07-01',
        ':endDate': '2008-06-30',
      },
    };

    let bucket = 'sharecast-exports';
    let exportName = '2007-2008';

    let onScan = function(err, data) {
      if (err) {
        console.log(err, err.stack);
      } else {
        for (let idx = 0; idx < data.Items.length; idx++) {
          csvData.push(data.Items[idx]);
          // Ensure we have the field name for the csv export
          Object.keys(data.Items[idx]).forEach((fieldName) => {
            if (csvFields.indexOf(fieldName) === -1) {
              csvFields.push(fieldName);
            }
          });
        }
        // If length has changed and it's a multiple of 1000, log it
        if (Math.floor(csvData.length / 1000) !== numRecs) {
          console.log(csvData.length + ' records...');
        }

        numRecs = Math.floor(csvData.length / 1000);
        if (typeof data.LastEvaluatedKey !== 'undefined') {
          params.ExclusiveStartKey = data.LastEvaluatedKey;
          dynamoDb.scan(params, onScan);
        } else {
          writeCsv(csvData, csvFields, bucket, exportName);
          console.log(`Writing export data`);
        }
      }
    };

    // describe the table and write metadata to the backup
    let table = awaitify(describeTable(tableName));

    // limit the the number or reads to match our capacity
    params.Limit = table.ProvisionedThroughput.ReadCapacityUnits;

    console.log(`Ready to start export scan`);

    // start streaming table data
    dynamoDb.scan(params, onScan);
  } catch (err) {
    console.error(err);
  }
});

let describeTable = function(tableName) {
  return new Promise(function(resolve, reject) {
    dynamo.describeTable({
      TableName: tableName,
    }, function(err, data) {
      if (err) {
        reject(err);
      } else {
        resolve(data.Table);
      }
    });
  });
};


module.exports.exportHandler = exportHandler;

exportHandler();
