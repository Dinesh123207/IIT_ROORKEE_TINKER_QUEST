import React from "react";
import "./style.css";
import Header from "../Header";
import Modal from "../UserModal";
import AdminModal from "../AdminModal";
import Footer from "../Footer";
import useStoryContext from "../../hooks/useProductContext";
import TextAnimate from "./textAnimate";
import "aos/dist/aos.css";
import Aos from "aos";
import Loader from "../Loader";
import { motion } from "framer-motion";

export default function Index() {
  let { modalPop, adminPop } = useStoryContext();
  const [isLoading, setIsLoading] = React.useState(true);

  React.useEffect(() => {
    Aos.init({ duration: 4000 });
  }, []);

  React.useEffect(() => {
    setTimeout(() => {
      setIsLoading(false);
    }, 2000);
  }, []);

  if (isLoading) {
    // add loader animation here
    return <Loader />;
  }

  return (
    <>
      <div className="home-page">
        <Header />
        <section id="home" class="bg-white dark:bg-gray-900 shadow-xl home">
          <div class="grid max-w-screen-xl px-4 py-8 mx-auto lg:gap-8 xl:gap-0 lg:py-16 lg:grid-cols-12">
            <div class="mr-auto place-self-center lg:col-span-7">
              {/* <TextAnimate /> */}
              <h1 class="max-w-2xl mb-4 text-4xl font-extrabold tracking-tight leading-none md:text-5xl xl:text-6xl dark:text-white">
                {/* Payments tool for software companies */}
                Ensuring your health today, for a secure tomorrow
              </h1>
              <p class="max-w-2xl mb-6 font-bold text-gray-500 lg:mb-8 md:text-lg lg:text-xl dark:text-gray-400">
                "Welcome to MedSecure -- where coverage meets care"
              </p>

              <div className="flex items-center">
                <p className="max-w-2xl font-normal text-gray-500  md:text-sm lg:text-base dark:text-gray-400">
                  Problem Statement Partner 
                </p>
                <img src="bajaj.png" alt="bajaj-logo" className="h-14" />
              </div>
            </div>
            <div class="hidden lg:mt-0 lg:col-span-5 lg:flex my-auto">
              <img
                className="rounded-xl border"
                src="shakthi.jpg"
                alt="mockup"
              />
            </div>
          </div>
        </section>

        <section
          id="features"
          className="bg-purple-100  dark:bg-gray-900"
          style={{
            backgroundImage: "url('bg1.png')",
            backgroundRepeat: "no-repeat",
            backgroundPositionX: "50%",
          }}
        >
          <div className="rounded-2xl max-w-screen-xl mx-auto px-4 py-16 format lg:format-lg dark:format-invert format-h2:mb-12">
            {/* <h2 className="">Steps to apply</h2> */}
            <h2>
              Unified Claims Experience
              <br /> Platform
            </h2>
            <div className="grid grid-cols-2 gap-24 items-center">
              <motion.div
                initial={{ scale: 0 }}
                whileInView={{ scale: 1 }}
                className="format bg-white dark:bg-primary-600 rounded-xl p-4 w-96 mx-auto relative left-20"
              >
                <div className="flex justify-between -mb-8">
                  {/* <h3>Step 2</h3> */}
                  <h3>Precision</h3>
                  <img src="ai.png" alt="" className="aspect-square w-14" />
                </div>
                <p>
                  Utilizing AI based Automated Document Verification ensuring
                  accuracy and efficiency in verification processes.
                </p>
              </motion.div>
              <motion.div
                initial={{ scale: 0 }}
                whileInView={{ scale: 1 }}
                className="format bg-white dark:bg-primary-600 rounded-xl p-4 w-96 mx-auto relative bottom-24 right-20"
              >
                <div className="flex justify-between -mb-8">
                  {/* <h3>Step 1</h3> */}
                  <h3> Seamless </h3>
                  <img
                    src="pattern.png"
                    alt=""
                    className="aspect-square w-12"
                  />
                </div>
                <p>
                  {/* Login with your credentials and apply claim with ease */}
                  Effortlessly apply claim by logging in with your credentials,
                  simplifying the process for a seamless experience.
                </p>
              </motion.div>
              <motion.div
                initial={{ scale: 0 }}
                whileInView={{ scale: 1 }}
                className="format bg-white dark:bg-primary-600 rounded-xl p-4 w-96 mx-auto relative left-20"
              >
                <div className="flex justify-between -mb-8">
                  {/* <h3>Step 4</h3> */}
                  <h3>Empower</h3>
                  <img
                    src="delivery.png"
                    alt=""
                    className="aspect-square w-12"
                  />
                </div>
                <p>
                  {/* Track Your Claim: Discover Your Status, Securely Download Verified Documents  */}
                  Stay informed and empowered - track your claim status and
                  securely access your verified documents with ease
                </p>
              </motion.div>
              <motion.div
                initial={{ scale: 0 }}
                whileInView={{ scale: 1 }}
                className="format bg-white dark:bg-primary-600 rounded-xl p-4 w-96 mx-auto relative right-20 bottom-24"
              >
                <div className="flex justify-between -mb-8">
                  {/* <h3>Step 3</h3> */}
                  <h3>Collaborate</h3>
                  <img
                    src="collaboration.png"
                    alt=""
                    className="aspect-square w-12"
                  />
                </div>
                <p>
                  Also augmenting our system with seasoned experts in document
                  verification
                </p>
              </motion.div>
            </div>
          </div>
        </section>
        <section class="bg-white dark:bg-gray-900">
          <div class="py-8 px-4 mx-auto max-w-screen-xl sm:py-16 lg:px-6">
            <div class="features max-w-screen-md mb-8 lg:mb-16">
              <h2 class="mb-4 text-4xl tracking-tight font-extrabold text-gray-900 dark:text-white flex items-center">
                Why choose MedSecure
                <img
                  src="logo2.png"
                  alt=""
                  class="aspect-square h-20 w-12 ml-5"
                />
              </h2>
            </div>
            <div class="space-y-8 md:grid md:grid-cols-2 lg:grid-cols-3 md:gap-12 md:space-y-0">
              {/* <div>
                <div class="flex justify-center items-center mb-4 w-10 h-10 rounded-full lg:h-12 lg:w-12  dark:bg-primary-900">
                   <img src="health-insurance.png" alt="" className="aspect-square  w-15" />
                </div>
                <h3 class="mb-2 text-xl font-bold dark:text-white">
                  Effortless Claim Apply
                </h3>
                <p class="text-gray-500 dark:text-gray-400">
                  Apply, verify, and track claims seamlessly through our advanced system, ensuring swift processing and hassle-free experience
                </p>
              </div> */}
              <motion.div
                initial={{ scaleX: 0 }}
                whileInView={{ scaleX: 1 }}
                class="custom-card  bg-primary-50 shadow-xl border"
              >
                <div class="flex justify-center items-center mb-4 w-10 h-10 rounded-full lg:h-12 lg:w-12  dark:bg-primary-900">
                  <img
                    src="health-insurance.png"
                    alt=""
                    class="aspect-square w-15"
                  />
                </div>
                <h3 class="mb-2 text-xl font-bold dark:text-white">
                  Effortless Claim Apply
                </h3>
                <p class="text-gray-500 dark:text-gray-400">
                  Apply, verify, and track claims seamlessly through our
                  advanced system, ensuring swift processing and hassle-free
                  experience
                </p>
              </motion.div>
              <motion.div
                initial={{ scaleX: 0 }}
                whileInView={{ scaleX: 1 }}
                class="custom-card  bg-primary-50 shadow-xl border"
              >
                <div class="flex justify-center items-center mb-4 w-10 h-10 rounded-full lg:h-12 lg:w-12">
                  <img
                    src="veracity.png"
                    alt=""
                    className="aspect-square  w-18"
                  />
                </div>
                <h3 class="mb-2 text-xl font-bold dark:text-white">
                  Precision and Trust
                </h3>
                <p class="text-gray-500 dark:text-gray-400">
                  Count on our AI model and expert-augmented verification for
                  accurate, reliable document verification, securing your
                  confidence
                </p>
              </motion.div>
              <motion.div
                initial={{ scaleX: 0 }}
                whileInView={{ scaleX: 1 }}
                class="custom-card bg-primary-50 shadow-xl border"
              >
                <div class="flex justify-center items-center mb-4 w-10 h-10 rounded-full lg:h-15 lg:w-12">
                  <img src="tracking.png" className="aspect-square w-30 h-45" />
                </div>
                <h3 class="mb-2 text-xl font-bold dark:text-white">
                  Empowering Your Journey
                </h3>
                <p class="text-gray-500 dark:text-gray-400">
                  Stay informed and in control with easy access to your claim
                  updates and verified documents, putting at the helm of your
                  healthcare
                </p>
              </motion.div>
            </div>
          </div>
        </section>
        <Footer />
      </div>
      {modalPop ? <Modal /> : null}
      {adminPop ? <AdminModal /> : null}
    </>
  );
}
