import React from "react";
import UserNav from "../UserDashboard/userNav";
import UserFooter from "../UserDashboard/userFooter";
import {BsArrowLeftShort, BsSearch , BsChevronDown} from "react-icons/bs";
import { AiFillEnvironment } from "react-icons/ai";
import {  AiOutlineInsurance, AiOutlineDashboard, AiOutlineUser } from 'react-icons/ai';
import { useState } from "react";
import { MdVerifiedUser } from "react-icons/md";
import { FcAcceptDatabase } from "react-icons/fc";
import { GiSuspicious } from "react-icons/gi";
import { MdCancel } from "react-icons/md";
import { MdOutlineDashboard } from "react-icons/md";
import { MdDashboard } from "react-icons/md";
// import PieChart from 'react-simple-pie-chart';
// import { BarChart , PieChart } from "reaviz";
import { DonutLargeRounded } from "@mui/icons-material";
import AcceptedClaims from "../AdminAcceptedDocuments";
import SuspiciousClaims from "../AdminSuspiciousDocuments";
import RejectedClaims from "../AdminRejectedDocuments";
import AdminNav from "./adminNav";
import { PieChart } from '@mui/x-charts/PieChart';
import Typography from '@mui/material/Typography';

export default function AdminDashboard(){

    const data = [
        { key: 'IDS', data: 14 },
        { key: 'Malware', data: 5 },
        { key: 'DLP', data: 18 }
      ];

    const [selected, setSelected] = useState("Dashboard");

    const [open, setOpen] = useState(true);
    const [submenuOpen , setSubmenuOpen] = useState(false);
    const Menus = [
    { title: "Dashboard", icon: <AiOutlineDashboard /> },
    // { title: "Verified Claims", spacing: true, icon: <MdVerifiedUser/> },
    {title:"Accepted Claims" , icon:<FcAcceptDatabase /> ,spacing:true},
    {title:"Suspicious Claims" , icon:<GiSuspicious/>},
    {title:"Rejected Claims", icon:<MdCancel/>},
  ];

    const renderComponent = ()=>{
      if(selected==="Dashboard"){
        return(

          <div className="min-h-screen max-w-screen-lg mx-auto flex items-center justify-between">
           
          <div className="shadow-xl rounded-md p-4 flex">
          <Typography variant="h6" gutterBottom>
        Claims Overview
      </Typography>
          <PieChart
      colors={['#f21001', '#f79431', '#2ECC40']} // Use palette
      series={[
        {
          data: [
            { value: 25, color: '#2ECC40', label:"Accepted Claims" },
            // Add more data objects as needed
            { value: 10, color: '#f79431' , label:"Suspicious Claims" },
            { value: 15, color: '#f21001', label:"Rejected Claims" }, // Use color property
            
          ],
        },
      ]}
      width={500}
      height={200}
    />
</div>

      <div className=" shadow-xl rounded-md flex">
      <Typography variant="h6" gutterBottom>
        Hospitals Overview
      </Typography>
<PieChart
      colors={['#02b2af', '#b800d8', '#2e96ff','#60009b']} // Use palette
      series={[
        {
          data: [
            { value: 25, color: '#02b2af', label:"Apollo Hospitals" },
            // Add more data objects as needed
            { value: 10, color: '#b800d8' , label:"Chettinad Hospital" },
            { value: 15, color: '#2e96ff', label:"Max Healthcare" }, // Use color property
            {value:20 , color:'#60009b' , label:"Manipal Hospital"}
            
          ],
        },
      ]}
      width={500}
      height={200}
    />

            {/* <PieChart
              width={350}
              height={400}
              data={data}
              className="bg-white rounded-md"
            />
            <PieChart
              width={350}
              height={400}
              data={data}
              className="bg-white rounded-md"
            /> */}
          </div>
        </div>
        //   <PieChart
        //   width={350}
        //   height={400}
        //   data={data}
        //   className="bg-white rounded-md"
        // />
        );
      }
      if(selected==="Accepted Claims"){
        return(
          <div>
            <AcceptedClaims/>
          </div>
        )
      }

      if(selected==="Suspicious Claims"){
        return(
          <div>
            <SuspiciousClaims/>
          </div>
        )
      }
      if(selected==="Rejected Claims"){
        return(
          <div>
            <RejectedClaims/>
          </div>
        )
      }
    };

    return(
     <div>
        <AdminNav/>

        <div className="flex">
           {/* Sidebar */}
      <div className="flex  ">
        {/* <div className={`bg-dark-purple h-screen p-5 pt-8 
      ${open?"w-72":"w-20"} duration-300 relative`}> */}
        <div
          className={`p-5 pt-8 bg-primary-800 sticky top-20 ${
            open ? "w-72" : "w-20"
          } duration-300 h-screen`}
        >
          <BsArrowLeftShort
            className={`bg-white text-dark-purple text-3xl rounded-full absolute
         -right-3 top-9 border-dark-purple 
         cursor-pointer ${!open && "rotate-180"} `}
            onClick={() => setOpen(!open)}
          />

          <div className="inline-flex">
            {/* <AiOutlineInsurance
              className={`bg-amber-300 text-4xl rounded cursor-pointer block float-left mr-2
          duration-500 ${open && "rotate-[360deg]"}
        `}
              style={{ fontSize: open ? "2rem" : "2.5rem" }}
            /> */}
            <MdOutlineDashboard className={` bg-white text-4xl rounded cursor-pointer block float-left mr-2
          duration-500 ${open && "rotate-[360deg]"}
        `} style={{ fontSize: open ? "2rem" : "2.5rem" }} />
            <h1
              className={`text-white origin-left font-medium text-2xl duration-300 ${
                !open && "scale-0"
              }`}
            >
              Hello, Admin
            </h1>
            <img
              src="/wave.gif"
              alt="wave"
              className={`w-8 h-8 ml-2 ${!open && "scale-0"}`}
              style={{ backgroundColor: "transparent" }}
            />
          </div>

          <ul className="pt-2">
            {Menus.map((menu, index) => (
              <li
                key={index}
                className={`text-gray-300 text-sm cursor-pointer p-2 hover:bg-light-white rounded-md ${
                  menu.spacing ? "mt-9" : "mt-2"
                } ${menu.title === selected ? "bg-red-400" : ""}}`}
              >
                <div
                  className="flex items-center gap-x-4"
                  onClick={() => {
                    if (menu.title === "Feedback and Information") {
                      setSubmenuOpen(!submenuOpen);
                      console.log(menu.title);
                    }
                    setSelected(menu.title);
                  }}
                >
                  <div className="text-2xl block float-left">
                    {menu.icon} {/* Display the menu item icon */}
                  </div>
                  <div
                    className={`text-base font-medium flex-1 duration-200 ${
                      !open && "hidden"
                    }`}
                  >
                    {menu.title}
                  </div>
                  {menu.submenu && open && (
                    <BsChevronDown
                      className={`${submenuOpen && "rotate-180"}`}
                    />
                  )}
                </div>
                {menu.submenu && submenuOpen && open && (
                  <ul className="bg-primary-700">
                    {menu.submenuItems.map((submenuItem, index) => (
                      <li
                        key={index}
                        className="text-gray-300 text-sm flex items-center
                  gap-x-4 cursor-pointer p-2 px-5 hover:bg-light-white rounded-md"
                      >
                        {submenuItem.icon} {submenuItem.title}
                      </li>
                    ))}
                  </ul>
                )}
                   </li>
            ))}
          </ul>

          </div>
        </div>

        <div className="p-8">{renderComponent()}</div>
      </div>

      <UserFooter/>
      </div>
    );
}

   




// import React from "react";
// import UserNav from "../UserDashboard/userNav";
// import UserFooter from "../UserDashboard/userFooter";
// import {BsArrowLeftShort, BsSearch , BsChevronDown} from "react-icons/bs";
// import { AiFillEnvironment } from "react-icons/ai";
// import {  AiOutlineInsurance, AiOutlineDashboard, AiOutlineUser } from 'react-icons/ai';
// import { useState } from "react";
// import { MdVerifiedUser } from "react-icons/md";
// import { FcAcceptDatabase } from "react-icons/fc";
// import { GiSuspicious } from "react-icons/gi";
// import { MdCancel } from "react-icons/md";
// import { MdOutlineDashboard } from "react-icons/md";
// import { MdDashboard } from "react-icons/md";
// // import PieChart from 'react-simple-pie-chart';
// import { BarChart , PieChart } from "reaviz";
// import { DonutLargeRounded } from "@mui/icons-material";
// import AcceptedClaims from "../AdminAcceptedDocuments";
// import SuspiciousClaims from "../AdminSuspiciousDocuments";
// import RejectedClaims from "../AdminRejectedDocuments";
// import AdminNav from "./adminNav";

// export default function AdminDashboard(){

//     const data = [
//         { key: 'IDS', data: 14 },
//         { key: 'Malware', data: 5 },
//         { key: 'DLP', data: 18 }
//       ];

//     const [selected, setSelected] = useState("Dashboard");

//     const [open, setOpen] = useState(true);
//     const [submenuOpen , setSubmenuOpen] = useState(false);
//     const Menus = [
//     { title: "Dashboard", icon: <AiOutlineDashboard /> },
//     { title: "Verified Claims", spacing: true, icon: <MdVerifiedUser/> },
//     {title:"Accepted Claims" , icon:<FcAcceptDatabase />},
//     {title:"Suspicious Claims" , icon:<GiSuspicious/>},
//     {title:"Rejected Claims", icon:<MdCancel/>},
//   ];

//     const renderComponent = ()=>{
//       if(selected==="Dashboard"){
//         return(
//           <PieChart
//           width={350}
//           height={400}
//           data={data}
//           className="bg-white rounded-md"
//         />
//         );
//       }
//       if(selected==="Accepted Claims"){
//         return(
//           <div>
//             <AcceptedClaims/>
//           </div>
//         )
//       }

//       if(selected==="Suspicious Claims"){
//         return(
//           <div>
//             <SuspiciousClaims/>
//           </div>
//         )
//       }
//       if(selected==="Rejected Claims"){
//         return(
//           <div>
//             <RejectedClaims/>
//           </div>
//         )
//       }
//     };

//     return(
//      <div>
//         <AdminNav/>

//         <div className="flex">
//            {/* Sidebar */}
//       <div className="flex  ">
//         {/* <div className={`bg-dark-purple h-screen p-5 pt-8 
//       ${open?"w-72":"w-20"} duration-300 relative`}> */}
//         <div
//           className={`p-5 pt-8 bg-primary-800 sticky top-20 ${
//             open ? "w-72" : "w-20"
//           } duration-300 h-screen`}
//         >
//           <BsArrowLeftShort
//             className={`bg-white text-dark-purple text-3xl rounded-full absolute
//          -right-3 top-9 border-dark-purple 
//          cursor-pointer ${!open && "rotate-180"} `}
//             onClick={() => setOpen(!open)}
//           />

//           <div className="inline-flex">
//             {/* <AiOutlineInsurance
//               className={`bg-amber-300 text-4xl rounded cursor-pointer block float-left mr-2
//           duration-500 ${open && "rotate-[360deg]"}
//         `}
//               style={{ fontSize: open ? "2rem" : "2.5rem" }}
//             /> */}
//             <MdOutlineDashboard className={` bg-white text-4xl rounded cursor-pointer block float-left mr-2
//           duration-500 ${open && "rotate-[360deg]"}
//         `} style={{ fontSize: open ? "2rem" : "2.5rem" }} />
//             <h1
//               className={`text-white origin-left font-medium text-2xl duration-300 ${
//                 !open && "scale-0"
//               }`}
//             >
//               Hello, Admin
//             </h1>
//             <img
//               src="wave.gif"
//               alt="wave"
//               className={`w-8 h-8 ml-2 ${!open && "scale-0"}`}
//               style={{ backgroundColor: "transparent" }}
//             />
//           </div>

//           <ul className="pt-2">
//             {Menus.map((menu, index) => (
//               <li
//                 key={index}
//                 className={`text-gray-300 text-sm cursor-pointer p-2 hover:bg-light-white rounded-md ${
//                   menu.spacing ? "mt-9" : "mt-2"
//                 } ${menu.title === selected ? "bg-red-400" : ""}}`}
//               >
//                 <div
//                   className="flex items-center gap-x-4"
//                   onClick={() => {
//                     if (menu.title === "Feedback and Information") {
//                       setSubmenuOpen(!submenuOpen);
//                       console.log(menu.title);
//                     }
//                     setSelected(menu.title);
//                   }}
//                 >
//                   <div className="text-2xl block float-left">
//                     {menu.icon} {/* Display the menu item icon */}
//                   </div>
//                   <div
//                     className={`text-base font-medium flex-1 duration-200 ${
//                       !open && "hidden"
//                     }`}
//                   >
//                     {menu.title}
//                   </div>
//                   {menu.submenu && open && (
//                     <BsChevronDown
//                       className={`${submenuOpen && "rotate-180"}`}
//                     />
//                   )}
//                 </div>
//                 {menu.submenu && submenuOpen && open && (
//                   <ul className="bg-primary-700">
//                     {menu.submenuItems.map((submenuItem, index) => (
//                       <li
//                         key={index}
//                         className="text-gray-300 text-sm flex items-center
//                   gap-x-4 cursor-pointer p-2 px-5 hover:bg-light-white rounded-md"
//                       >
//                         {submenuItem.icon} {submenuItem.title}
//                       </li>
//                     ))}
//                   </ul>
//                 )}
//                    </li>
//             ))}
//           </ul>

//           </div>
//         </div>

//         <div className="p-8">{renderComponent()}</div>
//       </div>

//       <UserFooter/>
//       </div>
//     );
// }

   


// import React from "react";
// import UserNav from "../UserDashboard/userNav";
// import UserFooter from "../UserDashboard/userFooter";
// import {BsArrowLeftShort, BsSearch , BsChevronDown} from "react-icons/bs";
// import { AiFillEnvironment } from "react-icons/ai";
// import {  AiOutlineInsurance, AiOutlineDashboard, AiOutlineUser } from 'react-icons/ai';
// import { useState } from "react";
// import { MdVerifiedUser } from "react-icons/md";
// import { FcAcceptDatabase } from "react-icons/fc";
// import { GiSuspicious } from "react-icons/gi";
// import { MdCancel } from "react-icons/md";
// import { MdOutlineDashboard } from "react-icons/md";
// import { MdDashboard } from "react-icons/md";
// // import PieChart from 'react-simple-pie-chart';
// import { BarChart , PieChart } from "reaviz";
// import { DonutLargeRounded } from "@mui/icons-material";
// import AcceptedClaims from "../AdminAcceptedDocuments";
// import SuspiciousClaims from "../AdminSuspiciousDocuments";
// import RejectedClaims from "../AdminRejectedDocuments";
// import AdminNav from "./adminNav";

// export default function AdminDashboard(){

//     const data = [
//         { key: 'IDS', data: 14 },
//         { key: 'Malware', data: 5 },
//         { key: 'DLP', data: 18 }
//       ];

//     const [selected, setSelected] = useState("Dashboard");

//     const [open, setOpen] = useState(true);
//     const [submenuOpen , setSubmenuOpen] = useState(false);
//     const Menus = [
//     { title: "Dashboard", icon: <AiOutlineDashboard /> },
//     { title: "Verified Claims", spacing: true, icon: <MdVerifiedUser/> },
//     {title:"Accepted Claims" , icon:<FcAcceptDatabase />},
//     {title:"Suspicious Claims" , icon:<GiSuspicious/>},
//     {title:"Rejected Claims", icon:<MdCancel/>},
//   ];

//     const renderComponent = ()=>{
//       if(selected==="Dashboard"){
//         return(
//           <PieChart
//           width={350}
//           height={400}
//           data={data}
//           className="bg-white rounded-md"
//         />
//         );
//       }
//       if(selected==="Accepted Claims"){
//         return(
//           <div>
//             <AcceptedClaims/>
//           </div>
//         )
//       }

//       if(selected==="Suspicious Claims"){
//         return(
//           <div>
//             <SuspiciousClaims/>
//           </div>
//         )
//       }
//       if(selected==="Rejected Claims"){
//         return(
//           <div>
//             <RejectedClaims/>
//           </div>
//         )
//       }
//     };

//     return(
//      <div>
//         <AdminNav/>

//         <div className="flex">
//            {/* Sidebar */}
//       <div className="flex  ">
//         {/* <div className={`bg-dark-purple h-screen p-5 pt-8 
//       ${open?"w-72":"w-20"} duration-300 relative`}> */}
//         <div
//           className={`p-5 pt-8 bg-primary-800 sticky top-20 ${
//             open ? "w-72" : "w-20"
//           } duration-300 h-screen`}
//         >
//           <BsArrowLeftShort
//             className={`bg-white text-dark-purple text-3xl rounded-full absolute
//          -right-3 top-9 border-dark-purple 
//          cursor-pointer ${!open && "rotate-180"} `}
//             onClick={() => setOpen(!open)}
//           />

//           <div className="inline-flex">
//             {/* <AiOutlineInsurance
//               className={`bg-amber-300 text-4xl rounded cursor-pointer block float-left mr-2
//           duration-500 ${open && "rotate-[360deg]"}
//         `}
//               style={{ fontSize: open ? "2rem" : "2.5rem" }}
//             /> */}
//             <MdOutlineDashboard className={` bg-white text-4xl rounded cursor-pointer block float-left mr-2
//           duration-500 ${open && "rotate-[360deg]"}
//         `} style={{ fontSize: open ? "2rem" : "2.5rem" }} />
//             <h1
//               className={`text-white origin-left font-medium text-2xl duration-300 ${
//                 !open && "scale-0"
//               }`}
//             >
//               Hello, Admin
//             </h1>
//             <img
//               src="wave.gif"
//               alt="wave"
//               className={`w-8 h-8 ml-2 ${!open && "scale-0"}`}
//               style={{ backgroundColor: "transparent" }}
//             />
//           </div>

//           <ul className="pt-2">
//             {Menus.map((menu, index) => (
//               <li
//                 key={index}
//                 className={`text-gray-300 text-sm cursor-pointer p-2 hover:bg-light-white rounded-md ${
//                   menu.spacing ? "mt-9" : "mt-2"
//                 } ${menu.title === selected ? "bg-red-400" : ""}}`}
//               >
//                 <div
//                   className="flex items-center gap-x-4"
//                   onClick={() => {
//                     if (menu.title === "Feedback and Information") {
//                       setSubmenuOpen(!submenuOpen);
//                       console.log(menu.title);
//                     }
//                     setSelected(menu.title);
//                   }}
//                 >
//                   <div className="text-2xl block float-left">
//                     {menu.icon} {/* Display the menu item icon */}
//                   </div>
//                   <div
//                     className={`text-base font-medium flex-1 duration-200 ${
//                       !open && "hidden"
//                     }`}
//                   >
//                     {menu.title}
//                   </div>
//                   {menu.submenu && open && (
//                     <BsChevronDown
//                       className={`${submenuOpen && "rotate-180"}`}
//                     />
//                   )}
//                 </div>
//                 {menu.submenu && submenuOpen && open && (
//                   <ul className="bg-primary-700">
//                     {menu.submenuItems.map((submenuItem, index) => (
//                       <li
//                         key={index}
//                         className="text-gray-300 text-sm flex items-center
//                   gap-x-4 cursor-pointer p-2 px-5 hover:bg-light-white rounded-md"
//                       >
//                         {submenuItem.icon} {submenuItem.title}
//                       </li>
//                     ))}
//                   </ul>
//                 )}
//                    </li>
//             ))}
//           </ul>

//           </div>
//         </div>

//         <div className="p-8">{renderComponent()}</div>
//       </div>

//       <UserFooter/>
//       </div>
//     );
// }

   