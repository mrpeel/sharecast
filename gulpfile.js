'use strict';

const gulp = require('gulp');
const install = require('gulp-install');
const zip = require('gulp-zip');
const del = require('del');
const awsLambda = require('gulp-aws-lambda');
const credentials = require('./credentials/aws.json');

const lambdaParamsRetrieveDaily = {
  FunctionName: 'retrieveShareData',
  Handler: 'index.dailyHandler',
  Role: 'arn:aws:iam::815588223950:role/lambda_write_dynamo',
  Runtime: 'nodejs4.3',
  Description: 'Retrieve share data and store in dynamodb',
  MemorySize: 1024,
  Timeout: 300,
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
  Runtime: 'nodejs4.3',
  Description: 'Retrieve share data and store in dynamodb',
  MemorySize: 1024,
  Timeout: 300,
  Publish: true,
  Code: {
    S3Bucket: 'cake-lambda-zips',
    S3Key: 'retrieve-share-data.zip',
  },
};

const lambdaParamsRetrieveAdjustedPrices = {
  FunctionName: 'retrieveAdjustedHistoryData',
  Handler: 'index.retrieveAdjustedHistoryDataHandler',
  Role: 'arn:aws:iam::815588223950:role/lambda_write_dynamo',
  Runtime: 'nodejs4.3',
  Description: 'Retrieve share data and store in dynamodb',
  MemorySize: 1024,
  Timeout: 300,
  Publish: true,
  Code: {
    S3Bucket: 'cake-lambda-zips',
    S3Key: 'retrieve-share-data.zip',
  },
};

const lambdaParamsProcessReturns = {
  FunctionName: 'processReturns',
  Handler: 'index.processReturnsHandler',
  Role: 'arn:aws:iam::815588223950:role/lambda_write_dynamo',
  Runtime: 'nodejs4.3',
  Description: 'Retrieve share data and store in dynamodb',
  MemorySize: 1024,
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

gulp.task('rootjs', ['clean'], function() {
  return gulp.src(['./lambda-handler/index.js',
    './lambda-handler/publish-sns.js'])
    .pipe(gulp.dest('dist/'));
});

gulp.task('retrievejs', ['rootjs'], function() {
  return gulp.src(['./retrieve-data/*.js'])
    .pipe(gulp.dest('dist/retrieve'));
});

gulp.task('credentials', ['retrievejs'], function() {
  return gulp.src(['./credentials/credentials.json'])
    .pipe(gulp.dest('dist/credentials'));
});


gulp.task('install_dependencies', ['credentials'], function() {
  return gulp.src('./package.json')
    .pipe(gulp.dest('./dist'))
    .pipe(install({
      production: true,
    }));
});

gulp.task('deploy', ['install_dependencies'], function() {
  gulp.src(['dist/**/*'])
    .pipe(zip('daily-retrieve-share-data.zip'))
    .pipe(awsLambda(awsCredentials, lambdaParamsRetrieveDaily));

  gulp.src(['dist/**/*'])
    .pipe(zip('check-adjusted-data.zip'))
    .pipe(awsLambda(awsCredentials, lambdaParamsCheckAdjustedPrices));

  gulp.src(['dist/**/*'])
    .pipe(zip('retrieve-adjusted-data.zip'))
    .pipe(awsLambda(awsCredentials, lambdaParamsRetrieveAdjustedPrices));

  gulp.src(['dist/**/*'])
    .pipe(zip('process-returns.zip'))
    .pipe(awsLambda(awsCredentials, lambdaParamsProcessReturns));
});
