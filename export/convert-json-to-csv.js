'use strict';

const aws = require('aws-sdk');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const json2csv = require('json2csv');

aws.config.update({
  region: 'ap-southeast-2',
});

aws.config.loadFromPath('../credentials/aws.json');

let retrieveJSONFile = function(bucket, fileName) {
  return new Promise(function(resolve, reject) {
    let s3obj = new aws.S3();
    let fileData = '';

    /* s3obj.getObject({
      Bucket: bucket,
      Key: fileName,
    }, (err, data) => {
      // Handle any error and exit
      if (err) {
        reject(err);
      }

      resolve(data.Body);
    });
  }); */

    /* s3obj.getObject({
      Bucket: bucket,
      Key: fileName,
    },
      function(err, data) {
        fileData = fileData + data.Body.toString();
      }).on('end', function() {
      resolve(fileData);
    }); */

    /* let appendStream = function(inStream) {
      console.log(inStream);
      fileData = fileData + inStream.data.Body.toString();
    };

    s3obj.getObject({
      Bucket: bucket,
      Key: fileName,
    }).createReadStream().pipe(appendStream)
      .on('end', function() {
        resolve(fileData);
      }); */


    s3obj.getObject({
      Bucket: bucket,
      Key: fileName,
    }).on('httpData', function(chunk) {
      fileData = fileData + chunk.toString();
    }).on('httpDone', function() {
      resolve(fileData);
    }).send();
  });
};

let convertJSONToCSV = asyncify(function() {
  let bucket = 'sharecast-exports';
  let fieldsFile = '2007-20082017-04-03-fields.json';
  let dataFile = '2007-20082017-04-03.json';
  let csvFile = '2007-2008-2017-04-03.csv';

  let fieldsSring = awaitify(retrieveJSONFile(bucket, fieldsFile));
  let jsonFields = JSON.parse(fieldsSring);
  jsonFields.csvFields.sort();

  let dataString = awaitify(retrieveJSONFile(bucket, dataFile));
  let jsonData = JSON.parse(dataString);

  let csvOutput = json2csv({
    data: jsonData.exportData,
    fields: jsonFields.csvFields,
  });

  let s3obj = new aws.S3({
    params: {
      Bucket: bucket,
      Key: csvFile,
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
});

convertJSONToCSV();
