'use strict';

const aws = require('aws-sdk');
const ReadableStream = require('./readable-stream');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');

aws.config.update({
  region: 'ap-southeast-2',
});

aws.config.loadFromPath('../credentials/aws.json');

let dynamo = new aws.DynamoDB();
let dynamoDb = new aws.DynamoDB.DocumentClient();

let writeJSONFile = function(jsonData, bucket, fileName) {
  let s3obj = new aws.S3({
    params: {
      Bucket: bucket,
      Key: fileName,
    },
  });

  s3obj.upload({
    Body: JSON.stringify(jsonData),
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

let exportHandler = asyncify(function() {
  try {
    let tableName = 'companyQuotes';
    let dataStream = new ReadableStream();
    let numRecs = 0;
    let newNumRecs = 0;
    let started = false;
    let csvFields = [];

    let today = new Date();
    let dateString = today.getFullYear() + '-' +
    ('0' + (today.getMonth() + 1)).slice(-2) + '-' +
    ('0' + today.getDate()).slice(-2);

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
    let exportName = '2007-2008' + dateString + '.json';
    let exportCSVName = '2007-2008' + dateString + '-fields.json';
    let body = dataStream;

    let s3obj = new aws.S3({
      params: {
        Bucket: bucket,
        Key: exportName,
      },
    });

    s3obj.upload({
      Body: body,
    }, function(err) {
      if (err) {
        console.error(err);
      } else {
        console.log('All done.');
      }
    }).on('httpUploadProgress', function(evt) {
      console.log('Uploading...');
    });

    let onScan = function(err, data) {
      if (err) {
        console.log(err, err.stack);
      } else {
        for (let idx = 0; idx < data.Items.length; idx++) {
          if (started) {
            dataStream.append(',');
          }
          dataStream.append('\n');
          dataStream.append(JSON.stringify(data.Items[idx]));
          started = true;

          // Ensure we have the field name for the csv export
          Object.keys(data.Items[idx]).forEach((fieldName) => {
            if (csvFields.indexOf(fieldName) === -1) {
              csvFields.push(fieldName);
            }
          });
        }

        newNumRecs = newNumRecs + data.Items.length;
        // If length has changed and it's a multiple of 1000, log it
        if (Math.floor(newNumRecs / 1000) !== numRecs) {
          console.log(newNumRecs + ' records...');
        }

        numRecs = Math.floor(newNumRecs / 1000);
        if (typeof data.LastEvaluatedKey !== 'undefined') {
          params.ExclusiveStartKey = data.LastEvaluatedKey;
          dynamoDb.scan(params, onScan);
        } else {
          dataStream.append('\n');
          dataStream.append(']}');
          dataStream.end();
          console.log(`Finalising export data stream`);
          writeJSONFile({
            csvFields: csvFields,
          }, bucket, exportCSVName);
        }
      }
    };

    // describe the table and write metadata to the backup
    let table = awaitify(describeTable(tableName));

    // limit the the number or reads to match our capacity
    params.Limit = table.ProvisionedThroughput.ReadCapacityUnits;

    console.log(`Ready to start export scan`);

    dataStream.append('{exportData: [');
    dataStream.append('\n');

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
