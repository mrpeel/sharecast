'use strict';

let aws = require('aws-sdk');
// let stream = require('stream');
let ReadableStream = require('./readable-stream');
let zlib = require('zlib');
let async = require('async');

let dateFormat = require('dateformat');
let ts = dateFormat(new Date(), 'yyyymmdd-HHMMss');

aws.config.update({
  region: 'ap-southeast-2',
});

aws.config.loadFromPath('../credentials/aws.json');

let dynamo = new aws.DynamoDB();

let backupTable = function(tablename, callback) {
  let dataStream = new ReadableStream(); // new stream.Readable();
  let gzip = zlib.createGzip();

  // create parameters hash for table scan
  let params = {
    TableName: tablename,
    ReturnConsumedCapacity: 'NONE',
    Limit: '1',
  };

  // body will contain the compressed content to ship to s3
  let body = dataStream.pipe(gzip);

  let s3obj = new aws.S3({
    params: {
      Bucket: 'sharecast-files',
      Key: 'backup/' + tablename + '/' + tablename + '-' + ts + '.gz',
    },
  });
  s3obj.upload({
    Body: body,
  }).on('httpUploadProgress', function(evt) {
    console.log(evt);
  }).send(function(err, data) {
    console.log(err, data); callback();
  });

  let onScan = function(err, data) {
    if (err) console.log(err, err.stack);
    else {
      for (let idx = 0; idx < data.Items.length; idx++) {
        dataStream.append(JSON.stringify(data.Items[idx]));
        dataStream.append('\n');
      }

      if (typeof data.LastEvaluatedKey !== 'undefined') {
        params.ExclusiveStartKey = data.LastEvaluatedKey;
        dynamo.scan(params, onScan);
      } else {
        dataStream.end();
      }
    }
  };

  // describe the table and write metadata to the backup
  dynamo.describeTable({
    TableName: tablename,
  }, function(err, data) {
    if (err) console.log(err, err.stack);
    else {
      let table = data.Table;
      // Write table metadata to first line
      dataStream.append(JSON.stringify(table));
      dataStream.append('\n');

      // limit the the number or reads to match our capacity
      params.Limit = table.ProvisionedThroughput.ReadCapacityUnits;

      // start streaminf table data
      dynamo.scan(params, onScan);
    }
  });
};

let backupAll = function(context) {
  dynamo.listTables({}, function(err, data) {
    if (err) console.log(err, err.stack); // an error occurred
    else {
      // Temporarily re-set to one table
      data.TableNames = ['companies'];

      async.each(data.TableNames, function(table, callback) {
        console.log('Backing up ' + table);
        backupTable(table, callback);
      }, function(err) {
        if (err) {
          console.log('A table failed to process');
        } else {
          console.log('All tables have been processed successfully');
        }
        context.done(err);
      });
    }
  });
};

module.exports.backupAll = backupAll;
