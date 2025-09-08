"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import {
  Dialog,
  DialogPanel,
  Disclosure,
  DisclosureButton,
  DisclosurePanel,
  PopoverGroup,
} from "@headlessui/react";
import { Bars3Icon, XMarkIcon } from "@heroicons/react/24/outline";
import {
  ChevronDownIcon,
  PhoneIcon,
  PlayCircleIcon,
} from "@heroicons/react/20/solid";

const products = [
  { name: "Product 1", href: "#" },
  { name: "Product 2", href: "#" },
  { name: "Product 3", href: "#" },
];

const callsToAction = [
  { name: "Watch demo", href: "#", icon: PlayCircleIcon },
  { name: "Contact sales", href: "#", icon: PhoneIcon },
];

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();

  // Only show logo on login and signup pages
  const isAuthPage = pathname === "/login" || pathname === "/signup";

  return (
    <header className="bg-white cursor-pointer">
      <nav
        className="mx-auto flex items-center justify-between py-3 px-6 lg:px-8 container"
        aria-label="Global"
      >
        {/* Logo */}
        <div className="flex lg:flex-1">
          <a href="/" className="-m-1.5 p-1.5">
            <span className="sr-only">Your Company</span>
            <img
              src="/assets/MindHaven.png"
              alt="logo"
              width={250}
              height={1450}
            />
          </a>
        </div>

        {/* Only render nav/buttons if not on auth pages */}
        {!isAuthPage && (
          <>  
            {/* Mobile Menu Button */}
            <div className="flex lg:hidden">
              <button
                type="button"
                onClick={() => setMobileMenuOpen(true)}
                className="-m-2.5 inline-flex items-center justify-center rounded-md p-2.5 text-gray-700"
              >
                <span className="sr-only">Open main menu</span>
                <Bars3Icon aria-hidden="true" className="size-6" />
              </button>
            </div>

            {/* Desktop Navigation */}
            <PopoverGroup className="hidden lg:flex lg:gap-x-12">
              <a href="/home" className="rounded-md text-sm px-4 py-2 bg-amber-50 font-semibold text-gray-900 hover:bg-amber-300">
                Home
              </a>
              <a href="/book-appointment" className="rounded-md text-sm px-4 py-2 bg-amber-50 font-semibold text-gray-900 hover:bg-amber-300">
                Book Appointment
              </a>
              <a href="/about" className="rounded-md text-sm px-4 py-2 bg-amber-50 font-semibold text-gray-900 hover:bg-amber-300">
                About Us
              </a>
            </PopoverGroup>
          </>
        )}
      </nav>

      {/* Mobile Navigation Menu */}
      {!isAuthPage && (
        <Dialog
          open={mobileMenuOpen}
          onClose={setMobileMenuOpen}
          className="lg:hidden"
        >
          <div className="fixed inset-0 z-10" />
          <DialogPanel className="fixed inset-y-0 right-0 z-10 w-full overflow-y-auto bg-white px-6 py-6 sm:max-w-sm sm:ring-1 sm:ring-gray-900/10">
            <div className="flex items-center justify-between">
              <a href="/" className="-m-1.5 p-1.5">
                <span className="sr-only">Your Company</span>
                <img
                  alt="logo"
                  src="/assets/MindHaven.png"
                  className="h-8 w-auto"
                />
              </a>
              <button
                type="button"
                onClick={() => setMobileMenuOpen(false)}
                className="-m-2.5 rounded-md p-2.5 text-gray-700"
              >
                <span className="sr-only">Close menu</span>
                <XMarkIcon aria-hidden="true" className="size-6" />
              </button>
            </div>

            {/* Mobile Menu Links */}
            <div className="mt-6 flow-root">
              <div className="-my-6 divide-y divide-gray-500/10">
                <div className="space-y-2 py-6">
                  {/* Products Section */}
                  <Disclosure as="div" className="-mx-3">
                    <DisclosureButton className="group flex w-full items-center justify-between rounded-lg py-2 pr-3.5 pl-3 text-base font-semibold text-gray-900 hover:bg-gray-50">
                      Products
                      <ChevronDownIcon
                        aria-hidden="true"
                        className="size-5 flex-none group-data-open:rotate-180"
                      />
                    </DisclosureButton>
                    <DisclosurePanel className="mt-2 space-y-2">
                      {[...products, ...callsToAction].map((item) => (
                        <a
                          key={item.name}
                          href={item.href}
                          className="block rounded-lg py-2 pr-3 pl-6 text-sm font-semibold text-gray-900 hover:bg-gray-50"
                        >
                          {item.name}
                        </a>
                      ))}
                    </DisclosurePanel>
                  </Disclosure>

                  {/* Other Links */}
                  <a
                    href="#"
                    className="-mx-3 block rounded-lg px-3 py-2 text-base font-semibold text-gray-900 hover:bg-gray-50"
                  >
                    Features
                  </a>
                  <a
                    href="#"
                    className="-mx-3 block rounded-lg px-3 py-2 text-base font-semibold text-gray-900 hover:bg-gray-50"
                  >
                    Marketplace
                  </a>
                  <a
                    href="#"
                    className="-mx-3 block rounded-lg px-3 py-2 text-base font-semibold text-gray-900 hover:bg-gray-50"
                  >
                    Company
                  </a>
                </div>
              </div>
            </div>
          </DialogPanel>
        </Dialog>
      )}
    </header>
  );
}
