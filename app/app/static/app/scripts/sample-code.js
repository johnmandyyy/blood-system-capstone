try {

  axios.defaults.headers.common["X-CSRFToken"] =
    globalVariables.getCookie("csrftoken");
  const encrpytedForm = {
    data: globalVariables.encryptData({}),
  };

  await axios
    .post("", encrpytedForm)
    .then((response) => {
      console.log(response)
    })
    .catch((error) => {
      console.log(error);
    });
}
catch (error) { console.log(error) }


try {

  var response = await axios.get("");
  var encrypted = response.data;
  var decrypted = globalVariables.decryptData(encrypted.data);
  this.polesWithNoSchedule = JSON.parse(decrypted);

  response = await axios.get("");
  encrypted = response.data;
  decrypted = globalVariables.decryptData(encrypted.data);
  this.polesWithSchedule = JSON.parse(decrypted);
} catch (error) {
  console.log(error);
} 