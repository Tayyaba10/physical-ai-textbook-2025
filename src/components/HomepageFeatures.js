import React from 'react';
import clsx from 'clsx';
import styles from './index.module.css';

const FeatureList = [
  {
    title: 'ROS 2 Foundation',
    description: (
      <>
        Learn the fundamentals of ROS 2 Humble Hawksbill, the backbone of modern robotics systems.
        Master nodes, topics, services, and actions for robust robotic communication.
      </>
    ),
  },
  {
    title: 'AI-Native Integration',
    description: (
      <>
        Integrate NVIDIA Isaac Platform with GPU-accelerated AI for perception, planning, and control.
        Implement cutting-edge computer vision and language models for robotics.
      </>
    ),
  },
  {
    title: 'Vision-Language-Action',
    description: (
      <>
        Connect natural language understanding with visual perception and robotic action execution.
        Create robots that understand and respond to human commands naturally.
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        {/*<Svg className={styles.featureSvg} alt={title} />*/}
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}