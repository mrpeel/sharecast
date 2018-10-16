'use strict';

const gulp = require('gulp');
const install = require('gulp-install');
const zip = require('gulp-zip');
const del = require('del');
const awsLambda = require('gulp-aws-lambda');
const credentials = require('./credentials/aws.json');

// const lambdaParamsRetrieveDaily = {
//   FunctionName: 'retrieveShareData',
//   Handler: 'index.dailyHandler',
//   Role: 'arn:aws:iam::815588223950:role/lambda_write_dynamo',
//   Runtime: 'nodejs8.10 ',
//   Description: 'Retrieve share data and store in dynamodb',
//   MemorySize: 256,
//   Timeout: 300,
//   Publish: true,
//   Code: {
//     S3Bucket: 'cake-lambda-zips',
//     S3Key: 'retrieve-share-data.zip',
//   },
// };

const lambdaParamsRetrieveFunction = {
  FunctionName: 'retrieveFunction',
  Handler: 'index.retrievalHandler',
  Role: 'arn:aws:iam::815588223950:role/lambda_write_dynamo',
  Runtime: 'nodejs8.10 ',
  Description: 'Run single retrieve function for share data and store in dynamodb',
  MemorySize: 256,
  Timeout: 600,
  Publish: true,
  Code: {
    S3Bucket: 'cake-lambda-zips',
    S3Key: 'retrieve-share-data.zip',
  },
};


const lambdaParamsCheckAdjustedPrices = {
  FunctionName: 'checkForAdjustments',
  Handler: 'index.checkForAdjustmentsHandler',
  Role: 'arn:aws:iam::815588223950:role/lambda_write_dynamo',
  Runtime: 'nodejs8.10 ',
  Description: 'Check for prices which have been adjusted in the last 8 days',
  MemorySize: 256,
  Timeout: 300,
  Publish: true,
  Code: {
    S3Bucket: 'cake-lambda-zips',
    S3Key: 'retrieve-share-data.zip',
  },
};

const lambdaParamsReloadQuote = {
  FunctionName: 'reloadQuote',
  Handler: 'index.reloadQuote',
  Role: 'arn:aws:iam::815588223950:role/lambda_write_dynamo',
  Runtime: 'nodejs8.10 ',
  Description: 'Reload quotes for identified symbols and date periods',
  MemorySize: 256,
  Timeout: 300,
  Publish: true,
  Code: {
    S3Bucket: 'cake-lambda-zips',
    S3Key: 'retrieve-share-data.zip',
  },
};


const awsCredentials = {
  accessKeyId: credentials['accessKeyId'],
  secretAccessKey: credentials['secretAccessKey'],
  region: credentials['region'],
};

gulp.task('clean', function() {
  return del(['./dist/**/*']);
});

gulp.task('rootjs', gulp.series('clean', function() {
  return gulp.src(['./lambda-handler/index.js',
      './lambda-handler/publish-sns.js',
    ])
    .pipe(gulp.dest('dist/'));
}));

gulp.task('retrievejs', gulp.series('rootjs', function() {
  return gulp.src(['./retrieve-data/*.js*'])
    .pipe(gulp.dest('dist/retrieve'));
}));

gulp.task('credentials', gulp.series('retrievejs', function() {
  return gulp.src(['./credentials/credentials.json'])
    .pipe(gulp.dest('dist/credentials'));
}));


gulp.task('install_dependencies', gulp.series('credentials', function() {
  return gulp.src('./package.json')
    .pipe(gulp.dest('./dist'))
    .pipe(install({
      production: true,
    }));
}));

gulp.task('deploy', gulp.series('install_dependencies', function(done) {
  gulp.src(['dist/**/*'])
    .pipe(zip('retrieve-share-data.zip'))
    // .pipe(awsLambda(awsCredentials, lambdaParamsRetrieveDaily))
    .pipe(awsLambda(awsCredentials, lambdaParamsRetrieveFunction))
    .pipe(awsLambda(awsCredentials, lambdaParamsCheckAdjustedPrices))
    .pipe(awsLambda(awsCredentials, lambdaParamsReloadQuote));
  // .pipe(awsLambda(awsCredentials, lambdaParamsProcessReturns));

  // gulp.src(['dist/**/*'])
  //  .pipe(zip('check-adjusted-data.zip'))
  //  .pipe(awsLambda(awsCredentials, lambdaParamsCheckAdjustedPrices));

  // gulp.src(['dist/**/*'])
  //  .pipe(zip('retrieve-adjusted-data.zip'))
  //  .pipe(awsLambda(awsCredentials, lambdaParamsRetrieveAdjustedPrices));

  // gulp.src(['dist/**/*'])
  //  .pipe(zip('process-returns.zip'))
  //  .pipe(awsLambda(awsCredentials, lambdaParamsProcessReturns));

  done();
}));
