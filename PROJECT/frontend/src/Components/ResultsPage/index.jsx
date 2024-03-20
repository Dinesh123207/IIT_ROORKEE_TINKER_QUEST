import React, { useEffect, useState } from "react";
import "./style.css";
// import useStoryContext from "../../../hooks/useProductContext";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Loader from "../Loader";

export default function Index() {
  // const { selectedFilename } = useStoryContext();
  let naviagte = useNavigate(null);
  let [im1, setIm1] = useState("");
  let [im2, setIm2] = useState("");
  let [im3, setIm3] = useState("");
  let [im_total, setImTotal] = useState("");
  let [status, setStatus] = useState("");
  let checkForTc1 = (s1, ele1) => {
    if (s1 >= 9) {
      ele1.style.border = "0.5rem solid green";
      setIm1("/correct.png");
    } else if (s1 >= 5 && s1 <= 8) {
      ele1.style.border = "0.5rem solid orange";
      setIm1("/tick-orange.png");
    } else {
      ele1.style.border = "0.5rem solid red";
      setIm1("/delete.png");
    }
  };
  let checkForTc2 = (s2, ele2) => {
    if (s2 >= 9) {
      ele2.style.border = "0.5rem solid green";
      setIm2("/correct.png");
    } else if (s2 >= 5 && s2 <= 8) {
      ele2.style.border = "0.5rem solid orange";
      setIm2("/tick-orange.png");
    } else {
      ele2.style.border = "0.5rem solid red";
      setIm2("/delete.png");
    }
  };

  let checkForTc3 = (s3, ele3) => {
    if (s3 >= 9) {
      ele3.style.border = "0.5rem solid green";
      setIm3("/correct.png");
    } else if (s3 >= 5 && s3 <= 8) {
      ele3.style.border = "0.5rem solid orange";
      setIm3("/tick-orange.png");
    } else {
      ele3.style.border = "0.5rem solid red";
      setIm3("/delete.png");
    }
  };

  let checkForTotal = (total, ele_tot) => {
    if (total >= 80) {
      ele_tot.style.border = "0.5rem solid green";
      setImTotal("/correct.png");
    } else if (total >= 50 && total <= 79) {
      ele_tot.style.border = "0.5rem solid orange";
      setImTotal("/tick-orange.png");
    } else {
      ele_tot.style.border = "0.5rem solid red";
      setImTotal("/delete.png");
    }
  };

  let setTheStatusOfDocument = (total) => {
    if (total >= 80) setStatus("Accepted");
    else if (total >= 50 && total <= 79) setStatus("Suspicious");
    else setStatus("Rejected");
  };
  let onClickSubmitClaim = async () => {
    await axios
      .get(`${process.env.REACT_APP_BACKEND_URL}api/claim-the-document`, {
        params: {
          email: localStorage.getItem("user-email"),
          token: localStorage.getItem("token"),
          document_name: localStorage.getItem("selected-file-name"),
          client_id: localStorage.getItem("user-client-id"),
          claim_status: status,
        },
      })
      .then((response) => {})
      .catch((err) => {
        localStorage.setItem("error", err);
      });

    toast.success("Claimed!", {
      position: "top-right",
      autoClose: 5000,
      hideProgressBar: false,
      closeOnClick: true,
      pauseOnHover: true,
      draggable: true,
      progress: undefined,
      theme: "colored",
      style: {
        fontFamily: "DM Sans",
        fontSize: "2rem",
      },
    });
    let token = localStorage.getItem("token");
    let user_email = localStorage.getItem("user-email");
    localStorage.clear();
    localStorage.setItem("token", token);
    localStorage.setItem("user-email", user_email);
    naviagte("/user/dashboard");
  };
  let calculateRandomNumberForClientID = () => {
    let minm = 100000;
    let maxm = 999999;
    let client_id = Math.floor(Math.random() * (maxm - minm + 1)) + minm;
    if (!localStorage.getItem("user-client-id")) {
      localStorage.setItem("user-client-id", client_id);
    }
  };
  let [isLoading, setIsLoading] = React.useState(true);
  React.useEffect(()=>{
    setTimeout(()=>{
      setIsLoading(false);
    }, 4000);
  }, []);
  useEffect(() => {
    let s1 = Number(localStorage.getItem("s1"));
    let s2 = Number(localStorage.getItem("s2"));
    let s3 = Number(localStorage.getItem("s3"));
    let total = Number(localStorage.getItem("total"));
    let ele1 = document.getElementsByClassName("score-1-value-card")[0];
    let ele2 = document.getElementsByClassName("score-2-value-card")[0];
    let ele3 = document.getElementsByClassName("score-3-value-card")[0];
    let ele_tot = document.getElementsByClassName("final-score-value-card")[0];

    checkForTc1(s1, ele1);
    checkForTc2(s2, ele2);
    checkForTc3(s3, ele3);
    checkForTotal(total, ele_tot);
    setTheStatusOfDocument(total);
  }, []);

  if(isLoading){
    return <Loader/>
  }
  return (
    <div className="result-details">
      <div className="result-caption-navbar">
        <h2 className="client-id-result">Client-ID</h2>
        <h2 className="client-name-result">Client-Name</h2>
        <h2 className="client-document-result">Client-Document</h2>
        <h2 className="client-score-result">Client-Status</h2>
      </div>

      <div className="result-caption-details">
        <h2 className="enter-client-id">
          {localStorage.getItem("user-client-id")}
        </h2>
        <h2 className="enter-client-name">
          {localStorage.getItem("user-name")}
        </h2>
        <h2 className="enter-client-document">
          {localStorage.getItem("selected-file-name")}
        </h2>
        <h2 className="enter-client-score">{status}</h2>
      </div>

      <div className="score-result-card">
        <div className="wrapper-score-result-card">
          <div className="score-result-left">
            <h1 className="scores-result-value">Results:</h1>
          </div>
          <div className="score-result-mid">
            <div className="score-1-value-card">
              <p className="score-1-value">
                <b>Test-criteria 1:</b> {localStorage.getItem("s1")}/10
              </p>
              <img src={im1} alt="" className="image-icon-score" />
            </div>
            <div className="score-2-value-card">
              <p className="score-2-value">
                <b>Test-criteria 2:</b> {localStorage.getItem("s2")}/10
              </p>
              <img src={im2} alt="" className="image-icon-score" />
            </div>
            <div className="score-3-value-card">
              <p className="score-3-value">
                <b>Test-criteria 3:</b> {localStorage.getItem("s3")}/10
              </p>
              <img src={im3} alt="" className="image-icon-score" />
            </div>

            <div className="final-score-value-card">
              <p className="score-3-value">
                <b>Final Result:</b> {localStorage.getItem("total")}/100
              </p>
              <img src={im_total} alt="" className="image-icon-score" />
            </div>
          </div>
          <div className="claim-and-download-report-result-right">
            <button className="claim-btn-result" onClick={onClickSubmitClaim}>
              Claim
            </button>
            <ToastContainer />
            <div className="download-btn-report-result">
              <img
                width="24"
                height="24"
                src="https://img.icons8.com/material-rounded/24/download--v1.png"
                alt="download-icon"
                className="download-btn-icon"
              />
              <a
                className="download-btn-caption"
                href={`${
                  process.env.REACT_APP_ENGINE_URL
                }static/${localStorage.getItem("image1")}`}
                download="report"
                target="_blank"
              >
                Download Report
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
