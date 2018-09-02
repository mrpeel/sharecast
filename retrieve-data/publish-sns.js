'use strict';

const aws = require('aws-sdk');

aws.config.update({
  region: 'ap-southeast-2',
});

const sns = new aws.SNS();

let publishMsg = async function(topicArn, message, subject) {
  let params = {
    TopicArn: topicArn,
    Message: message,
  };
  if (subject) {
    params.Subject = subject;
  }

  try {
    let data = await sns.publish(params).promise();
    return data.MessageId;
  } catch (err) {
    throw err;
  }
};

module.exports = {
  publishMsg: publishMsg,
};
