export default function Footer() {
  return (
    <footer className="footer">
      <div className="container p-6">
        <div className="grid grid-cols-1 md:grid-cols-4">
          <div className="md:col-span-2">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 bg-primary-100 rounded-xl flex items-center justify-center">
                <span className="text-white font-bold text-lg">🛡️</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Spam Detection</h3>
                <p className="text-sm text-gray-400">AI-Powered Security</p>
              </div>
            </div>
            <p className="text-gray-300 text-sm max-w-md">
              Advanced AI-powered spam detection using machine learning algorithms 
              to protect your digital communications with high accuracy and real-time analysis.
            </p>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-6 text-white">Features</h3>
            <ul className="text-gray-300 text-sm space-y-3">
              <li className="flex items-center">
                <span className="w-2 h-2 bg-green-500 rounded-full mr-3"></span>
                Real-time spam detection
              </li>
              <li className="flex items-center">
                <span className="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
                High accuracy classification
              </li>
              <li className="flex items-center">
                <span className="w-2 h-2 bg-purple-500 rounded-full mr-3"></span>
                Secure API endpoints
              </li>
              <li className="flex items-center">
                <span className="w-2 h-2 bg-yellow-500 rounded-full mr-3"></span>
                Easy integration
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-6 text-white">Technology</h3>
            <ul className="text-gray-300 text-sm space-y-3">
              <li className="flex items-center">
                <span className="w-2 h-2 bg-orange-500 rounded-full mr-3"></span>
                Next.js & TypeScript
              </li>
              <li className="flex items-center">
                <span className="w-2 h-2 bg-green-500 rounded-full mr-3"></span>
                Machine Learning Models
              </li>
              <li className="flex items-center">
                <span className="w-2 h-2 bg-red-500 rounded-full mr-3"></span>
                Natural Language Processing
              </li>
              <li className="flex items-center">
                <span className="w-2 h-2 bg-yellow-500 rounded-full mr-3"></span>
                RESTful API
              </li>
            </ul>
          </div>
        </div>
        <div className="border-t border-gray-700 mt-12 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400 text-sm mb-4 md:mb-0">
              © 2024 Spam Detection App. Built for Computing Technology Project Assignment.
            </p>
            <div className="flex space-x-6 text-sm text-gray-400">
              <a href="#" className="hover:text-white">Privacy</a>
              <a href="#" className="hover:text-white">Terms</a>
              <a href="#" className="hover:text-white">Contact</a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}
