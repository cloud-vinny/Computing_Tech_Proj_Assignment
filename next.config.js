/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://computingtechprojassignment-production.up.railway.app/:path*',
      },
    ]
  },
}

module.exports = nextConfig
