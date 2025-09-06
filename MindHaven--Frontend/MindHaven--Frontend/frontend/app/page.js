import Banner from "./_components/Home/Banner";
import Footer from "./_components/Home/Footer";
export default function Home() {
  return (
    <>
      <div className="container mx-auto p-6 lg:px-8">
        <Banner /> 
        <Footer />
      </div>
    </>
  );
};