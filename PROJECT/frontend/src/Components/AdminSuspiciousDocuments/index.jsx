import React, { useEffect, useState } from "react";
import "./style.css";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

export default function SuspiciousClaims() {
  let [docs, setDocs] = useState([]);
  let [clickDropdown, setClickDropdown] = useState(false);
  let [selectedClientId, setSelectedClientId] = useState();
  let [feedbackBoxContent, setFeedbackBoxContent] = useState("");
  let getAllSuspiciousDocs = async () => {
    await axios
      .get(
        `${process.env.REACT_APP_BACKEND_URL}api/get-admin-documents/Suspicious`,
        {
          params: {
            token: localStorage.getItem("token"),
            email: localStorage.getItem("user-email"),
          },
        }
      )
      .then((response) => {
        setDocs(response.data.result);
      })
      .catch((err) => {
        localStorage.setItem("error", err);
      });
  };

  let changeDocumentStatus = async (id, curr_status, user_email) => {
    await axios
      .get(
        `${process.env.REACT_APP_BACKEND_URL}api/change-admin-document-status`,
        {
          params: {
            token: localStorage.getItem("token"),
            admin_email: localStorage.getItem("user-email"),
            user_email: user_email,
            feedback: feedbackBoxContent,
            curr_status: curr_status,
            client_id: id,
          },
        }
      )
      .then((response) => {
        if (response.data.status === 200) {
          getAllSuspiciousDocs();
        } else {
          console.log(response.data);
        }
      })
      .catch((err) => {
        localStorage.setItem("error", err);
      });
  };

  let onClickSubmitFeedback = (id) => {
    if (clickDropdown === true && selectedClientId === id) {
      setClickDropdown(false);
    } else {
      setSelectedClientId(id);
      setClickDropdown(true);
    }
  };

  let onClickRejectDocument = (id, user_email) => {
    toast.warn(`The document with client-id: ${id} is Rejected!`, {
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
        backgroundColor: "red",
        color: "white",
      },
    });
    changeDocumentStatus(id, "Rejected", user_email);
  };

  let onClickAcceptDocument = (id, user_email) => {
    toast.success(`The document with client-id: ${id} is Accepted!`, {
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
        backgroundColor: "green",
        color: "white",
      },
    });
    changeDocumentStatus(id, "Accepted", user_email);
  };

  useEffect(() => {
    getAllSuspiciousDocs();
  }, []);
  return (
    <div className="document-content-suspicious">
      <div className="document-caption-navbar">
        <h5 className="client-id-document">Client-ID</h5>
        <h5 className="client-name-document">Client-Name</h5>
        <h5 className="client-document-document">Client-Document</h5>
        <h5 className="client-score-document">Client-Status</h5>
      </div>
      <div className="document-wrapper-card-box">
        {/* <div className="document-caption-details-card" onClick={onClickSubmitFeedback}>
          <div className="wrapper-document-up">
            <h2 className="document-client-id">123456</h2>
            <h2 className="document-client-name">aditya</h2>
            <h2 className="document-client-document">a1.png</h2>
            <h2 className="document-client-score">
              <b>Suspicious</b>
            </h2>
          </div>
          {clickDropdown && <div className="wrapper-document-down">
            <textarea
              name="feedback-document"
              className="feedback-document-textarea"
              cols="30"
              rows="3"
              placeholder="Enter your Feedback..."
            ></textarea>
            <button className="reject-btn-document">Reject</button>
            <button className="accept-btn-document">Accept</button>
          </div>}
        </div> */}
        {docs.length >= 1 &&
          docs.map((doc) => (
            <div className="document-caption-details-card" key={doc.client_id}>
              <div
                className="wrapper-document-up"
                onClick={() => onClickSubmitFeedback(doc.client_id)}
              >
                <h2 className="document-client-id">{doc.client_id}</h2>
                <h2 className="document-client-name">{doc.name}</h2>
                <h2 className="document-client-document">
                  {doc.document_name}
                </h2>
                <h2 className="document-client-score">
                  <b>{doc.claim_status}</b>
                </h2>
              </div>
              {clickDropdown && selectedClientId === doc.client_id && (
                <div className="wrapper-document-down">
                  <div className="contact-detials-client">
                    <p className="email-id-client">
                      <b>Email - ID:</b> {doc.email}
                    </p>
                    <p className="mobile-number-client">
                      <b>Mobile No.:</b> {doc.mobile}
                    </p>
                  </div>
                  <div className="document-download-btn-report-result">
                    <img
                      width="24"
                      height="24"
                      src="https://img.icons8.com/windows/32/FFFFFF/download--v1.png"
                      alt="download-icon"
                      className="document-download-btn-icon"
                    />
                    <a
                      className="document-download-btn-caption"
                      href={`${
                        process.env.REACT_APP_ENGINE_URL
                      }static/${doc.client_id}-${doc.document_name}`}
                      download="report"
                      target="_blank"
                    >
                      Download Document
                    </a>
                  </div>
                  <div className="report-download-btn-report-result">
                    <img
                      width="24"
                      height="24"
                      src="https://img.icons8.com/windows/32/FFFFFF/download--v1.png"
                      alt="download-icon"
                      className="document-download-btn-icon"
                    />
                    <a
                      className="report-download-btn-caption"
                      href={`${
                        process.env.REACT_APP_ENGINE_URL
                      }static/${doc.client_id}-processed.png`}
                      download="report"
                      target="_blank"
                    >
                      Download Report
                    </a>
                  </div>
                  <textarea
                    name="feedback-document"
                    className="feedback-document-textarea"
                    cols="30"
                    rows="2"
                    placeholder="Enter your Feedback..."
                    onChange={(e) => setFeedbackBoxContent(e.target.value)}
                    required
                  ></textarea>
                  <button
                    className="reject-btn-document"
                    onClick={() =>
                      onClickRejectDocument(doc.client_id, doc.email)
                    }
                  >
                    Reject
                  </button>
                  <button
                    className="accept-btn-document"
                    onClick={() =>
                      onClickAcceptDocument(doc.client_id, doc.email)
                    }
                  >
                    Accept
                  </button>
                </div>
              )}
            </div>
          ))}
      </div>
    </div>
  );
}
