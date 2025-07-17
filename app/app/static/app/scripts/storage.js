const globalVariables = new Vue({
  data: {
    Bearer: null,
  },
  methods: {
    getCookie(name) {
      const cookieValue = document.cookie
        .split(";")
        .map((cookie) => cookie.trim())
        .find((cookie) => cookie.startsWith(name + "="));

      if (cookieValue) {
        return decodeURIComponent(cookieValue.split("=")[1]);
      } else {
        return null;
      }
    },
    encryptData(unencryptedData) {
      const encrypted = AES256.encrypt(
        JSON.stringify(unencryptedData),
        "SECRET-KEY"
      );
      return encrypted;
    },
    decryptData(encrypted) {
      return AES256.decrypt(encrypted, "SECRET-KEY");
    },
    convertImageToBase64(imageFile) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = function (event) {
          const base64String = event.target.result.split(",")[1]; // Extract the Base64 data part
          resolve(base64String);
        };

        reader.onerror = function (error) {
          reject(error);
        };

        reader.readAsDataURL(imageFile);
      });
    },
  },
});

