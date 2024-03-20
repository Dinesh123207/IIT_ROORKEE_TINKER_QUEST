import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import UserNav from "../UserDashboard/userNav";
import UserFooter from "../UserDashboard/userFooter";
import "./style.css";

const MultipleFileUpload = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [file, setFile] = useState();

  const onDrop = useCallback((acceptedFiles) => {
    setSelectedFiles(acceptedFiles);
  }, []);

  const handleChangeFile = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const response = await axios.post(
        "http://127.0.0.1:5002/upload1",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      console.log(response.data);
    } catch (error) {
      console.error("Error uploading files:", error);
    }
  };

  const handleSubmitFile = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", file);
    //engine api call
    try {
      const response = await axios.post(
        "http://127.0.0.1:5002/upload2",
        formData
      );
      let data = response.data;
      localStorage.setItem("json", JSON.stringify(data));
      if (response.status === 200) {
        console.log(data);
      } else {
        // Handle the error response
        localStorage.setItem("res", "try more :(");
      }
    } catch (error) {
      localStorage.setItem("error", error);
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    directory: true,
  });

  return (
    <>
    <UserNav/>
    <div className="page-container">
    <div className="pt-50 flex items-center justify-center py-4 md:py-60 flex-wrap" >
    <div className=" justify-items-center p-8 grid grid-cols-2 md:grid-cols-2 gap-20">
       
      <form onSubmit={handleSubmit} className="w-full"  style={{
            backdropFilter: "none",
            borderRadius: "10px", 
            backgroundColor: "#FFFFFF", 
            padding: "20px", 
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)", 
            width: "400px", 
    height: "170px", 
  }}>
        <div
          {...getRootProps()}
          style={{
            border: "2px dashed #eee",
            padding: "20px",
            textAlign: "center",
            cursor: "pointer",
            boxShadow:"1px solid rgba(0, 0, 0, 0.1)",
            backdropFilter: "none",
          }}
        >
          <input {...getInputProps()} />
          <p>Drag and drop a folder here, or click to select a folder</p>
        </div>
        <button type="submit">Upload</button>
      </form>
      <form onSubmit={handleSubmitFile} className="w-full" style={{
            backdropFilter: "none",
            borderRadius: "10px", 
            backgroundColor: "#FFFFFF", 
            padding: "20px", 
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)", 
            width: "400px", 
            height: "170px", 
  }}>
        <div
          style={{
            border: "2px dashed #eee",
            padding: "20px",
            textAlign: "center",
            cursor: "pointer",
            boxShadow:"1px solid rgba(0, 0, 0, 0.1)",
            backdropFilter: "none",
          }}
        >
          <input type="file" name="file" onChange={handleChangeFile} />
          <p>Drag and drop a folder here, or click to select a folder</p>
        </div>
        <button type="submit">Upload</button>
      </form>
      </div>
      </div></div>
      <UserFooter/>
    
    </>
  );
};

export default MultipleFileUpload;
