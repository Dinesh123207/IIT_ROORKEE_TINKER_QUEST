
import React, { useEffect, useState } from "react";
import "./style.css";
import axios from "axios";

export default function ClaimHistory() {
  let [docs, setDocs] = useState([]);
  let [clickDropdown, setClickDropdown] = useState(false);
  let [selectedClientId, setSelectedClientId] = useState();

  let getAllSuspiciousDocs = async () => {
    await axios
      .get(`${process.env.REACT_APP_BACKEND_URL}api/user-claim-history`, {
        params: {
          token: localStorage.getItem("token"),
          email: localStorage.getItem("user-email"),
        },
      })
      .then((response) => {
        setDocs(response.data.history);
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

  let changeColorsForDocumentStatus = () => {
    try{
      let ele1 = document.getElementsByClassName("user-history-client-score")[0].textContent;
      let ele2 = document.getElementsByClassName("user-history-client-score")[0];
      if (ele1 === "Accepted"){
        ele2.style = "color: green";
      }
      else if (ele1 === "Rejected"){
        ele2.style = `color: red`;
      }
    }
    catch(err){
      console.log(err);
    }
  }

  useEffect(() => {
    getAllSuspiciousDocs();
    changeColorsForDocumentStatus();
  }, []);
  return (
    
    <div className="user-history-content-accepted">
      <div className="user-history-caption-navbar ">
        <h5 className="client-id-user-history">Client-ID</h5>
        <h5 className="client-name-user-history">Client-Name</h5>
        <h5 className="client-document-user-history">Client-Document</h5>
        <h5 className="client-score-user-history">Client-Status</h5>
      </div>
      <div className="user-history-wrapper-card-box">
        {docs.length >= 1 &&
          docs.map((doc) => (
            <div
              className="user-history-caption-details-card"
              key={doc.client_id}
            >
              <div
                className="wrapper-user-history-up"
                onClick={() => onClickSubmitFeedback(doc.client_id)}
              >
                <h2 className="user-history-client-id">{doc.client_id}</h2>
                <h2 className="user-history-client-name">{doc.name}</h2>
                <h2 className="user-history-client-document">
                  {doc.document_name}
                </h2>
                <h2 className={`user-history-client-score user-history-client-score-${doc.claim_status}`}>
                  <b>{doc.claim_status}</b>
                </h2>
              </div>
              {clickDropdown && selectedClientId === doc.client_id && (
                <div className="wrapper-user-history-down">
                  <div className="contact-detials-client">
                    <p className="email-id-client">
                      <b>Email - ID:</b> {doc.email}
                    </p>
                    <p className="mobile-number-client">
                      <b>Mobile No.:</b> {doc.mobile}
                    </p>
                  </div>
                  <div className="document-download-btn-report-result">
                    {/* <img
                      width="24"
                      height="24"
                      src="https://img.icons8.com/windows/32/FFFFFF/download--v1.png"
                      alt="download-icon"
                      className="document-download-btn-icon"
                    /> */}
                    <a
                      className="document-download-btn-caption"
                      href={`${process.env.REACT_APP_ENGINE_URL}static/${doc.client_id}-${doc.document_name}`}
                      download="report"
                      target="_blank"
                    >
                    Download Document
                    </a>
                  </div>
                  <div className="report-download-btn-report-result">
                    {/* <img
                      width="24"
                      height="24"
                      src="https://img.icons8.com/windows/32/FFFFFF/download--v1.png"
                      alt="download-icon"
                      className="document-download-btn-icon"
                    /> */}
                    <a
                      className="report-download-btn-caption"
                      href={`${process.env.REACT_APP_ENGINE_URL}static/${doc.client_id}-processed.png`}
                      download="report"
                      target="_blank"
                    >
                     Download Report
                    </a>
                  </div>
                  {doc.feedback && (
                    <p className="user-history-feedback-from-admin">
                      <b>Feedback(from admin): </b> &nbsp; "{doc.feedback}"
                    </p>
                  )}
                </div>
              )}
            </div>
          ))}
      </div>
      </div>

  );
}



// import React, { useEffect, useState } from "react";
// import "./style.css";
// import axios from "axios";

// export default function ClaimHistory() {
//   let [docs, setDocs] = useState([]);
//   let [clickDropdown, setClickDropdown] = useState(false);
//   let [selectedClientId, setSelectedClientId] = useState();

//   let getAllSuspiciousDocs = async () => {
//     await axios
//       .get(`${process.env.REACT_APP_BACKEND_URL}api/user-claim-history`, {
//         params: {
//           token: localStorage.getItem("token"),
//           email: localStorage.getItem("user-email"),
//         },
//       })
//       .then((response) => {
//         setDocs(response.data.history);
//       })
//       .catch((err) => {
//         localStorage.setItem("error", err);
//       });
//   };

//   let onClickSubmitFeedback = (id) => {
//     if (clickDropdown === true && selectedClientId === id) {
//       setClickDropdown(false);
//     } else {
//       setSelectedClientId(id);
//       setClickDropdown(true);
//     }
//   };

//   let changeColorsForDocumentStatus = () => {
//     try{
//       let ele1 = document.getElementsByClassName("user-history-client-score")[0].textContent;
//       let ele2 = document.getElementsByClassName("user-history-client-score")[0];
//       if (ele1 === "Accepted"){
//         ele2.style = "color: green";
//       }
//       else if (ele1 === "Rejected"){
//         ele2.style = `color: red`;
//       }
//     }
//     catch(err){
//       console.log(err);
//     }
//   }

//   useEffect(() => {
//     getAllSuspiciousDocs();
//     changeColorsForDocumentStatus();
//   }, []);
//   return (
//     <div className="user-history-content-accepted">
//       <div className="user-history-caption-navbar">
//         <h2 className="client-id-user-history">Client-ID</h2>
//         <h2 className="client-name-user-history">Client-Name</h2>
//         <h2 className="client-document-user-history">Client-Document</h2>
//         <h2 className="client-score-user-history">Client-Status</h2>
//       </div>
//       <div className="user-history-wrapper-card-box">
//         {docs.length >= 1 &&
//           docs.map((doc) => (
//             <div
//               className="user-history-caption-details-card"
//               key={doc.client_id}
//             >
//               <div
//                 className="wrapper-user-history-up"
//                 onClick={() => onClickSubmitFeedback(doc.client_id)}
//               >
//                 <h2 className="user-history-client-id">{doc.client_id}</h2>
//                 <h2 className="user-history-client-name">{doc.name}</h2>
//                 <h2 className="user-history-client-document">
//                   {doc.document_name}
//                 </h2>
//                 <h2 className={`user-history-client-score user-history-client-score-${doc.claim_status}`}>
//                   <b>{doc.claim_status}</b>
//                 </h2>
//               </div>
//               {clickDropdown && selectedClientId === doc.client_id && (
//                 <div className="wrapper-user-history-down">
//                   <div className="contact-detials-client">
//                     <p className="email-id-client">
//                       <b>Email - ID:</b> {doc.email}
//                     </p>
//                     <p className="mobile-number-client">
//                       <b>Mobile No.:</b> {doc.mobile}
//                     </p>
//                   </div>
//                   <div className="document-download-btn-report-result">
//                     <img
//                       width="24"
//                       height="24"
//                       src="https://img.icons8.com/windows/32/FFFFFF/download--v1.png"
//                       alt="download-icon"
//                       className="document-download-btn-icon"
//                     />
//                     <a
//                       className="document-download-btn-caption"
//                       href={`${process.env.REACT_APP_ENGINE_URL}static/${doc.client_id}-${doc.document_name}`}
//                       download="report"
//                       target="_blank"
//                     >
//                       Download Document
//                     </a>
//                   </div>
//                   <div className="report-download-btn-report-result">
//                     <img
//                       width="24"
//                       height="24"
//                       src="https://img.icons8.com/windows/32/FFFFFF/download--v1.png"
//                       alt="download-icon"
//                       className="document-download-btn-icon"
//                     />
//                     <a
//                       className="report-download-btn-caption"
//                       href={`${process.env.REACT_APP_ENGINE_URL}static/${doc.client_id}-processed.png`}
//                       download="report"
//                       target="_blank"
//                     >
//                       Download Report
//                     </a>
//                   </div>
//                   {doc.feedback && (
//                     <p className="user-history-feedback-from-admin">
//                       <b>Feedback(from admin): </b> &nbsp; "{doc.feedback}"
//                     </p>
//                   )}
//                 </div>
//               )}
//             </div>
//           ))}
//       </div>
//     </div>
//   );
// }

