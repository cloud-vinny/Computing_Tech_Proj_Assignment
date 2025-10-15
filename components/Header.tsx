export default function Header() {
  return (
    <header className="header">
      <div className="container">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-primary-100 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-lg">🛡️</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Spam Detection</h1>
              <p className="text-sm text-gray-600">AI-Powered Security</p>
            </div>
          </div>
          <nav className="hidden md:flex space-x-8">
            <a href="#" className="text-gray-600 font-semibold">Home</a>
            <a href="#" className="text-gray-600 font-semibold">About</a>
            <a href="#" className="text-gray-600 font-semibold">API</a>
            <a href="#" className="btn btn-primary">
              Get Started
            </a>
          </nav>
        </div>
      </div>
    </header>
  )
}
