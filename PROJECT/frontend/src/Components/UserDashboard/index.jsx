import React from "react";
import { BsArrowLeftShort, BsSearch, BsChevronDown } from "react-icons/bs";
import { AiFillEnvironment } from "react-icons/ai";
import { MdInsertChartOutlined , MdOutlineDashboard} from "react-icons/md";
import {
  AiOutlineDashboard,
  AiOutlineUser,
  AiOutlineFileDone,
  AiOutlineHistory,
  AiOutlineInsurance,
  AiOutlineSetting,
  AiOutlineLogout,
} from "react-icons/ai";
import { useState } from "react";
import UserNav from "./userNav";
import UserFooter from "./userFooter";
// import { PieChart } from "reaviz";
import { VscFeedback } from "react-icons/vsc";
import { LuFiles } from "react-icons/lu";
import { AiOutlineQuestionCircle } from "react-icons/ai";
import { MonetizationOnRounded } from "@mui/icons-material";
import { Healing } from "@mui/icons-material";
import ApplyClaim from "../UserApplyClaim";
import ClaimHistory from "../UserClaimHistory";
import { MdDashboard } from "react-icons/md";
import { PieChart } from '@mui/x-charts/PieChart';

export default function UserDashboard() {
  // const data = [
  //   { key: "Accepted Claims", data: 14 , color: "#4CAF50"},
  //   { key: "Suspicious Claims", data: 5 , color: "#FFC107"},
  //   { key: "Rejected Claims", data: 18 ,color: "#F44336"},
  // ];
  // const customColors =['#FF4136', '#FFC300', '#2ECC40'];
  // which option is selected
  const [selected, setSelected] = useState("Dashboard");

  const [open, setOpen] = useState(true);
  const [submenuOpen, setSubmenuOpen] = useState(false);
  const Menus = [
    { title: "Dashboard", icon: <AiOutlineDashboard /> },
    // { title: "User Details", icon: <AiOutlineUser /> },
    { title: "Apply Claim", spacing: true, icon: <AiOutlineFileDone /> },
    // { title: "Claim Status", icon: <MdInsertChartOutlined /> },
    { title: "Claim History", icon: <AiOutlineHistory /> },
    { title: "Renewal Insurance", icon: <AiOutlineInsurance /> },
    {
      title: "Feedback and Information",
      spacing: true,
      submenu: true,
      icon: <VscFeedback />,
      submenuItems: [
        { title: "Terms, Policies and Licenses", icon: <LuFiles /> },
        { title: "Browse FAQs", icon: <AiOutlineQuestionCircle /> },
      ],
    },
  ];

  const renderComponent = () => {
    if (selected === "Dashboard") {
      return (
       
        <div className="min-h-screen max-w-screen-lg mx-auto flex items-center justify-between">
          <div className="shadow-xl rounded-md p-4 flex">
          {/* <PieChart
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
    /> */}
          {/* <PieChart
      series={[
        {
          data: [
            { id: 0, value: 10, label: 'series A' },
            { id: 1, value: 15, label: 'series B' },
            { id: 2, value: 20, label: 'series C' },
          ],
          colorScheme: {customColors},
        },
      ]}
      width={400}
      height={200}
    /> */}

      {/* {data.map((item, index) => (
        <PieChart
          key={index}
          width={350}
          height={400}
          data={[item]}
          options={{
            slices: {
              [item.key]: { color: item.color },
            },
          }}
          className="bg-white rounded-md"
        />
      ))} */}
       {/* {data.map((item, index) => (
        <PieChart
          key={index}
          width={350}
          height={400}
          data={[item]}
          options={{
            slices: {
              [item.key]: { color: item.color },
            },
          }}
          className="bg-white rounded-md"
        />
      ))} */}
  </div>
</div>

        // <div className="min-h-screen max-w-screen-lg mx-auto flex items-center justify-between">
        // <PieChart
        //   width={350}
        //   height={400}
        //   data={data}
        //   className="bg-white rounded-md"
        // />
        // <PieChart
        //   width={350}
        //   height={400}
        //   data={data}
        //   className="bg-white rounded-md"
        // />
        // </div>
      );
    }
    if (selected === "Apply Claim") {
      return (
        
          <ApplyClaim/>
      );
    }

    if(selected==="Claim History"){
      return(
        <div>
          <ClaimHistory/>
          </div>
      )
    }
  };

  return (
    <div>
      <UserNav />
      {/* Sidebar */}
      <div className="flex h-screen grid-cols-12">
        <div
          className={`p-5 pt-8 bg-primary-800 sticky top-0 ${
            open ? "w-96" : "w-20"
          } duration-300 h-screen col-span-2 z-50`}
        >
          <BsArrowLeftShort
            className={`bg-white text-dark-purple text-3xl rounded-full absolute
         -right-3 top-9 border-dark-purple 
         cursor-pointer ${!open && "rotate-180"} `}
            onClick={() => setOpen(!open)}
          />

          <div className="inline-flex">
            <MdOutlineDashboard
              className={`bg-white text-4xl rounded cursor-pointer block float-left mr-2
          duration-500 ${open && "rotate-[360deg]"}
        `}
              style={{ fontSize: open ? "2rem" : "2.5rem" }}
            />
            <h1
              className={`text-white origin-left font-medium text-2xl duration-300 ${
                !open && "scale-0"
              }`}
            >
              Hello, User
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

        {/* Main content of Dashboard, components are rendered here based on option selected */}
        <div className="p-8 w-full" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>{renderComponent()}</div>
      </div>

      <UserFooter />
    </div>
  );
}


// import React from "react";
// import { BsArrowLeftShort, BsSearch, BsChevronDown } from "react-icons/bs";
// import { AiFillEnvironment } from "react-icons/ai";
// import { MdInsertChartOutlined } from "react-icons/md";
// import {
//   AiOutlineDashboard,
//   AiOutlineUser,
//   AiOutlineFileDone,
//   AiOutlineHistory,
//   AiOutlineInsurance,
//   AiOutlineSetting,
//   AiOutlineLogout,
// } from "react-icons/ai";
// import { useState } from "react";
// import UserNav from "./userNav";
// import UserFooter from "./userFooter";
// import { PieChart } from "reaviz";
// import { VscFeedback } from "react-icons/vsc";
// import { LuFiles } from "react-icons/lu";
// import { AiOutlineQuestionCircle } from "react-icons/ai";
// import { MonetizationOnRounded } from "@mui/icons-material";
// import { Healing } from "@mui/icons-material";
// import ApplyClaim from "../UserApplyClaim";
// import ClaimHistory from "../UserClaimHistory";

// export default function UserDashboard() {
//   const data = [
//     { key: "IDS", data: 14 },
//     { key: "Malware", data: 5 },
//     { key: "DLP", data: 18 },
//   ];

//   // which option is selected
//   const [selected, setSelected] = useState("Dashboard");

//   const [open, setOpen] = useState(true);
//   const [submenuOpen, setSubmenuOpen] = useState(false);
//   const Menus = [
//     { title: "Dashboard", icon: <AiOutlineDashboard /> },
//     // { title: "User Details", icon: <AiOutlineUser /> },
//     { title: "Apply Claim", spacing: true, icon: <AiOutlineFileDone /> },
//     // { title: "Claim Status", icon: <MdInsertChartOutlined /> },
//     { title: "Claim History", icon: <AiOutlineHistory /> },
//     { title: "Renewal Insurance", icon: <AiOutlineInsurance /> },
//     {
//       title: "Feedback and Information",
//       spacing: true,
//       submenu: true,
//       icon: <VscFeedback />,
//       submenuItems: [
//         { title: "Terms, Policies and Licenses", icon: <LuFiles /> },
//         { title: "Browse FAQs", icon: <AiOutlineQuestionCircle /> },
//       ],
//     },
//   ];

//   const renderComponent = () => {
//     if (selected === "Dashboard") {
//       return (
//         <PieChart
//           width={350}
//           height={400}
//           data={data}
//           className="bg-white rounded-md"
//         />
//       );
//     }
//     if (selected === "Apply Claim") {
//       return (
//         <div>
//           <ApplyClaim/>
//         </div>
//       );
//     }

//     if(selected==="Claim History"){
//       return(
//         <div>
//           <ClaimHistory/>
//           </div>
//       )
//     }
//   };

//   return (
//     <div>
//       <UserNav />

//       {/* Sidebar */}
//       <div className="flex  ">
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
//             <AiOutlineInsurance
//               className={`bg-amber-300 text-4xl rounded cursor-pointer block float-left mr-2
//           duration-500 ${open && "rotate-[360deg]"}
//         `}
//               style={{ fontSize: open ? "2rem" : "2.5rem" }}
//             />
//             <h1
//               className={`text-white origin-left font-medium text-2xl duration-300 ${
//                 !open && "scale-0"
//               }`}
//             >
//               Hello, User
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
               
//               </li>
//             ))}
//           </ul>

//         </div>

//         {/* Main content of Dashboard, components are rendered here based on option selected */}
//         <div className="p-8">{renderComponent()}</div>
//       </div>

//       <UserFooter />
//     </div>
//   );
// }






// import React from "react";
// import { BsArrowLeftShort, BsSearch, BsChevronDown } from "react-icons/bs";
// import { AiFillEnvironment } from "react-icons/ai";
// import { MdInsertChartOutlined } from "react-icons/md";
// import {
//   AiOutlineDashboard,
//   AiOutlineUser,
//   AiOutlineFileDone,
//   AiOutlineHistory,
//   AiOutlineInsurance,
//   AiOutlineSetting,
//   AiOutlineLogout,
// } from "react-icons/ai";
// import { useState } from "react";
// import UserNav from "./userNav";
// import UserFooter from "./userFooter";
// import { PieChart } from "reaviz";
// import { VscFeedback } from "react-icons/vsc";
// import { LuFiles } from "react-icons/lu";
// import { AiOutlineQuestionCircle } from "react-icons/ai";
// import { MonetizationOnRounded } from "@mui/icons-material";
// import { Healing } from "@mui/icons-material";
// import ApplyClaim from "../UserApplyClaim";
// import ClaimHistory from "../UserClaimHistory";

// export default function UserDashboard() {
//   const data = [
//     { key: "IDS", data: 14 },
//     { key: "Malware", data: 5 },
//     { key: "DLP", data: 18 },
//   ];

//   // which option is selected
//   const [selected, setSelected] = useState("Dashboard");

//   const [open, setOpen] = useState(true);
//   const [submenuOpen, setSubmenuOpen] = useState(false);
//   const Menus = [
//     { title: "Dashboard", icon: <AiOutlineDashboard /> },
//     // { title: "User Details", icon: <AiOutlineUser /> },
//     { title: "Apply Claim", spacing: true, icon: <AiOutlineFileDone /> },
//     // { title: "Claim Status", icon: <MdInsertChartOutlined /> },
//     { title: "Claim History", icon: <AiOutlineHistory /> },
//     { title: "Renewal Insurance", icon: <AiOutlineInsurance /> },
//     {
//       title: "Feedback and Information",
//       spacing: true,
//       submenu: true,
//       icon: <VscFeedback />,
//       submenuItems: [
//         { title: "Terms, Policies and Licenses", icon: <LuFiles /> },
//         { title: "Browse FAQs", icon: <AiOutlineQuestionCircle /> },
//       ],
//     },
//   ];

//   const renderComponent = () => {
//     if (selected === "Dashboard") {
//       return (
//         <PieChart
//           width={350}
//           height={400}
//           data={data}
//           className="bg-white rounded-md"
//         />
//       );
//     }
//     if (selected === "Apply Claim") {
//       return (
//         <div>
//           <ApplyClaim/>
//         </div>
//       );
//     }

//     if(selected==="Claim History"){
//       return(
//         <div>
//           <ClaimHistory/>
//           </div>
//       )
//     }
//   };

//   return (
//     <div>
//       <UserNav />

//       {/* Sidebar */}
//       <div className="flex  ">
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
//             <AiOutlineInsurance
//               className={`bg-amber-300 text-4xl rounded cursor-pointer block float-left mr-2
//           duration-500 ${open && "rotate-[360deg]"}
//         `}
//               style={{ fontSize: open ? "2rem" : "2.5rem" }}
//             />
//             <h1
//               className={`text-white origin-left font-medium text-2xl duration-300 ${
//                 !open && "scale-0"
//               }`}
//             >
//               Hello, User
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
               
//               </li>
//             ))}
//           </ul>

//         </div>

//         {/* Main content of Dashboard, components are rendered here based on option selected */}
//         <div className="p-8">{renderComponent()}</div>
//       </div>

//       <UserFooter />
//     </div>
//   );
// }








// import React from "react";
// import { BsArrowLeftShort, BsSearch, BsChevronDown } from "react-icons/bs";
// import { AiFillEnvironment } from "react-icons/ai";
// import { MdInsertChartOutlined } from "react-icons/md";
// import {
//   AiOutlineDashboard,
//   AiOutlineUser,
//   AiOutlineFileDone,
//   AiOutlineHistory,
//   AiOutlineInsurance,
//   AiOutlineSetting,
//   AiOutlineLogout,
// } from "react-icons/ai";
// import { useState } from "react";
// import UserNav from "./userNav";
// import UserFooter from "./userFooter";
// import { PieChart } from "reaviz";
// import { VscFeedback } from "react-icons/vsc";
// import { LuFiles } from "react-icons/lu";
// import { AiOutlineQuestionCircle } from "react-icons/ai";
// import { MonetizationOnRounded } from "@mui/icons-material";
// import { Healing } from "@mui/icons-material";
// import ApplyClaim from "../UserApplyClaim";
// import ClaimHistory from "../UserClaimHistory";

// export default function UserDashboard() {
//   const data = [
//     { key: "IDS", data: 14 },
//     { key: "Malware", data: 5 },
//     { key: "DLP", data: 18 },
//   ];

//   // which option is selected
//   const [selected, setSelected] = useState("Dashboard");

//   const [open, setOpen] = useState(true);
//   const [submenuOpen, setSubmenuOpen] = useState(false);
//   const Menus = [
//     { title: "Dashboard", icon: <AiOutlineDashboard /> },
//     // { title: "User Details", icon: <AiOutlineUser /> },
//     { title: "Apply Claim", spacing: true, icon: <AiOutlineFileDone /> },
//     // { title: "Claim Status", icon: <MdInsertChartOutlined /> },
//     { title: "Claim History", icon: <AiOutlineHistory /> },
//     { title: "Renewal Insurance", icon: <AiOutlineInsurance /> },
//     {
//       title: "Feedback and Information",
//       spacing: true,
//       submenu: true,
//       icon: <VscFeedback />,
//       submenuItems: [
//         { title: "Terms, Policies and Licenses", icon: <LuFiles /> },
//         { title: "Browse FAQs", icon: <AiOutlineQuestionCircle /> },
//       ],
//     },
//   ];

//   const renderComponent = () => {
//     if (selected === "Dashboard") {
//       return (
//         <PieChart
//           width={350}
//           height={400}
//           data={data}
//           className="bg-white rounded-md"
//         />
//       );
//     }
//     if (selected === "Apply Claim") {
//       return (
//         <div>
//           <ApplyClaim/>
//         </div>
//       );
//     }

//     if(selected==="Claim History"){
//       return(
//         <div>
//           <ClaimHistory/>
//           </div>
//       )
//     }
//   };

//   return (
//     <div>
//       <UserNav />

//       {/* Sidebar */}
//       <div className="flex  ">
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
//             <AiOutlineInsurance
//               className={`bg-amber-300 text-4xl rounded cursor-pointer block float-left mr-2
//           duration-500 ${open && "rotate-[360deg]"}
//         `}
//               style={{ fontSize: open ? "2rem" : "2.5rem" }}
//             />
//             <h1
//               className={`text-white origin-left font-medium text-2xl duration-300 ${
//                 !open && "scale-0"
//               }`}
//             >
//               Hello, User
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
               
//               </li>
//             ))}
//           </ul>

//         </div>

//         {/* Main content of Dashboard, components are rendered here based on option selected */}
//         <div className="p-8  w-full ">{renderComponent()}</div>
//       </div>

//       <UserFooter />
//     </div>
//   );
// }
