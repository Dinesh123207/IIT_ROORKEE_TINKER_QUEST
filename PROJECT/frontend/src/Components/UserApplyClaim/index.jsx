
import React, { useState } from "react";
import "./style.css";
import axios from "axios";
import { useNavigate } from "react-router-dom";

export default function ApplyClaim() {
  let navigate = useNavigate(null);
  const [file, setFile] = useState(null);
  const [image, setImage] = useState(null);
  const [processed, setProcessed] = useState(null);
  const [selectedFile, setSelectedFile] = useState(false);
  const [processedFile, setProcessedFile] = useState(false);
  let [name, setName] = useState("");
  let [mobile, setMobile] = useState("");

  let calculateRandomNumberForClientID = () => {
    let minm = 100000;
    let maxm = 999999;
    let client_id = Math.floor(Math.random() * (maxm - minm + 1)) + minm;
    if (!localStorage.getItem("user-client-id")) {
      localStorage.setItem("user-client-id", client_id);
    }
  };

  const handleChange = (e) => {
    setFile(e.target.files[0]);
    setSelectedFile(true);
    setProcessedFile(false);
    if (e.target.files && e.target.files[0]) {
      setImage(URL.createObjectURL(e.target.files[0]));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setProcessedFile(false);
    const formData = new FormData();
    formData.append("file", file);
    localStorage.setItem("selected-file-name", file.name);
    calculateRandomNumberForClientID();
    const flask_data = {
      client_id: localStorage.getItem("user-client-id"),
      file: file
    }
    //engine api call
    try {
      const response = await axios.post(
        `${process.env.REACT_APP_ENGINE_URL}upload`,
        formData
      );
      let data = response.data;
      localStorage.setItem("json", JSON.stringify(data));
      if (response.status === 200) {
        localStorage.setItem("s1", data.result1);
        localStorage.setItem("s2", data.result2);
        localStorage.setItem("s3", data.result3);
        localStorage.setItem("total", data.total);
        localStorage.setItem("image", data.image);
        localStorage.setItem("user-client-id", data.client_id);
        setProcessed(`${data.image}?${new Date().getTime()}`);
        setProcessedFile(true);
      } else {
        // Handle the error response
        localStorage.setItem("res", "try more :(");
      }
    } catch (error) {
      localStorage.setItem("error", error);
    }

    //backend api call
    localStorage.setItem("user-name", name);
    localStorage.setItem("user-mbl", mobile);

    await axios
      .get(`${process.env.REACT_APP_BACKEND_URL}api/get-user-claim-detail`, {
        params: {
          email: localStorage.getItem("user-email"),
          token: localStorage.getItem("token"),
          name: name,
          mobile: mobile,
        },
      })
      .then((response) => {

      })
      .catch((error) => {
        localStorage.setItem("error", error);
      });

    navigate("/user/result");
  };

  return (
    // <div className="background-image">
    <div className="max-w-screen-lg mx-auto p-8 shadow-lg rounded-lg custom-box">
    <div className="w-fit p-8 flex gap- rounded-xl mx-auto">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <h1 className="text-2xl font-bold ">Apply For the Claim</h1>
        <input
          type="text"
          className="w-96 px-6 py-3 rounded-xl outline-none border border-gray-300"
          name="name"
          placeholder="Enter your Full-Name"
          onChange={(e) => setName(e.target.value)}
          required
        />
        <input
          type="tel"
          name="mobile"
          className="w-96 px-6 py-3 rounded-xl outline-none border border-gray-300"
          placeholder="Enter your mobile number"
          onChange={(e) => setMobile(e.target.value)}
          required
        />
        <input
          type="email"
          name="email"
          className="w-96 px-6 py-3 rounded-xl outline-none border border-gray-300"
          placeholder="Enter your email"
          value={localStorage.getItem("user-email")}
          readOnly
        />

        <input
          onChange={handleChange}
          required
          class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400"
          type="file"
          name="file"
          id="file"
        />

        <button
          type="submit"
          className="bg-primary-700 rounded-xl py-4 px-2 text-lg font-medium tracking-wide text-white hover:bg-primary-600 transition-colors duration-200 ease-in-out justify-self-end"
        >
          Upload
        </button>
      </form>
      <div className="text-black">
        <div className="px-8">
          <h1 className="text-3xl font-bold tracking-wider my-4">MedSecure</h1>
          <p className="">
            <b>Fill the form</b>, and <b>upload your document</b> to file the
            claim!
          </p>
          <img src="/med2-right.png" alt="" className="w-96" />
        </div>
      </div>
    </div>
</div>
// </div>

  );
}
