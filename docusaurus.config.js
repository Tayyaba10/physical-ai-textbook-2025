module.exports = {
  title: 'AI-Native Robotics Textbook',
  tagline: 'From ROS 2 Foundation to Vision-Language-Action Integration',
  url: 'https://Tayyaba10.github.io',  // Replace with your actual domain
  baseUrl: '/physical-ai-textbook-2025/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',

  // GitHub pages deployment config.
  organizationName: 'Tayyaba10', // Usually your GitHub org/user name.
  projectName: 'physical-ai-textbook-2025', // Usually your repo name.
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/Tayyaba10/physical-ai-textbook-2025/edit/main/',
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/Tayyaba10/physical-ai-textbook-2025/edit/main/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'Physical-ai-textbook-2025',
      logo: {
        alt: 'Robotics Logo',
        src: 'img/robot-logo.svg',  // You can create this logo or use a placeholder
      },
      items: [
        {
          type: 'doc',
          docId: 'overview',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/Tayyaba10/physical-ai-textbook-2025',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Modules',
          items: [
            {
              label: 'Module 1: Robotic Nervous System',
              to: '/docs/module-1-robotic-nervous-system/',
            },
            {
              label: 'Module 2: Digital Twin & Simulation',
              to: '/docs/module-2-digital-twin/',
            },
            {
              label: 'Module 3: Isaac Platform & GPU AI',
              to: '/docs/module-3-ai-robot-brain/',
            },
            {
              label: 'Module 4: Vision-Language-Action Integration',
              to: '/docs/module-4-vision-language-action/',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/ros2',
            },
            {
              label: 'Robotics Stack Exchange',
              href: 'https://robotics.stackexchange.com/',
            },
            {
              label: 'ROS Discourse',
              href: 'https://discourse.ros.org/',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/Tayyyaba10/physical-ai-textbook-2025',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical Ai Textbook-2025. Built with Docusaurus.`,
    },
    prism: {
      theme: require('prism-react-renderer/themes/github'),
      darkTheme: require('prism-react-renderer/themes/dracula'),
      additionalLanguages: ['bash', 'json', 'python', 'cpp'],
    },
  },

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'module-1',
        path: 'docs/module-1-robotic-nervous-system',
        routeBasePath: 'docs/module-1-robotic-nervous-system',
        sidebarPath: require.resolve('./sidebars-module-1.js'),
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'module-2',
        path: 'docs/module-2-digital-twin',
        routeBasePath: 'docs/module-2-digital-twin',
        sidebarPath: require.resolve('./sidebars-module-2.js'),
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'module-3',
        path: 'docs/module-3-ai-robot-brain',
        routeBasePath: 'docs/module-3-ai-robot-brain',
        sidebarPath: require.resolve('./sidebars-module-3.js'),
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'module-4',
        path: 'docs/module-4-vision-language-action',
        routeBasePath: 'docs/module-4-vision-language-action',
        sidebarPath: require.resolve('./sidebars-module-4.js'),
      },
    ],
  ],
};