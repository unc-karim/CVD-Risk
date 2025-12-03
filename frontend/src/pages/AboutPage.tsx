/**
 * About page - Team, project information, and model descriptions
 */

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Grow,
  Container,
  Divider,
} from '@mui/material';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import TimelineIcon from '@mui/icons-material/Timeline';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import HubIcon from '@mui/icons-material/Hub';
import GroupIcon from '@mui/icons-material/Group';
import LanguageIcon from '@mui/icons-material/Language';

const AboutPage: React.FC = () => {
  const teamMembers = [
    {
      name: 'Karim Abdallah',
    },
    {
      name: 'Hussein Mdaihly',
    },
    {
      name: 'Hassan Hashem',
    },
    {
      name: 'Carl Wakim',
    },
  ];

  const models = [
    {
      title: 'Hypertension Detection',
      icon: FavoriteBorderIcon,
      color: '#ff6f61',
      bgColor: '#ffeaea',
      description: 'Detects signs of hypertensive retinopathy from fundus images',
      details: [
        'Uses RETFound Vision Transformer (ViT) architecture',
        'Binary classification: Hypertension Detected / Not Detected',
        'Provides probability score and confidence level',
        'Analyzes retinal blood vessel changes and microaneurysms',
        'Clinical significance: Early detection of hypertension-related eye damage',
      ],
    },
    {
      title: 'CIMT Regression',
      icon: TimelineIcon,
      color: '#64b5f6',
      bgColor: '#e3f2fd',
      description: 'Predicts carotid intima-media thickness for cardiovascular risk assessment',
      details: [
        'Uses Siamese Multimodal architecture',
        'Accepts bilateral carotid ultrasound images plus clinical data (age, sex)',
        'Predicts thickness measurement in millimeters (0.4 - 1.2mm)',
        'Categorizes risk level: Normal, Borderline, or Thickened',
        'Clinical significance: CIMT is a strong predictor of cardiovascular disease',
      ],
    },
    {
      title: 'A/V Segmentation',
      icon: AccountTreeIcon,
      color: '#81c784',
      bgColor: '#e8f5e9',
      description: 'Segments retinal blood vessels and extracts vascular metrics',
      details: [
        'Uses U-Net deep learning architecture',
        'Performs precise segmentation of retinal vessels in fundus images',
        'Extracts 6 vascular metrics:',
        '  • Vessel Density: Overall vessel coverage percentage',
        '  • Peripheral Density: Vessel coverage in outer retina',
        '  • Average Vessel Width: Mean diameter of detected vessels',
        '  • Fractal Dimension: Complexity and branching pattern',
        '  • Branching Density: Number of vessel branches',
        '  • Average Tortuosity: Vessel straightness/curviness',
        'Clinical significance: Vascular changes indicate systemic cardiovascular dysfunction',
      ],
    },
    {
      title: 'Fusion Model (CVD Risk)',
      icon: HubIcon,
      color: '#b39ddb',
      bgColor: '#f3e5f5',
      description: 'Comprehensive cardiovascular disease risk assessment combining all models',
      details: [
        'Uses Meta-Classifier (MLP) that fuses outputs from HTN, CIMT, and Vessel models',
        'Integrates bilateral fundus images and demographic data (age, sex)',
        'Produces overall CVD risk classification: Low, Medium, or High',
        'Generates CVD probability score',
        'Provides personalized recommendations based on contributing factors',
        'Clinical significance: Multimodal approach captures different aspects of cardiovascular health',
        'Offers more comprehensive risk assessment than individual models alone',
      ],
    },
  ];

  return (
    <Box
      sx={{
        width: '100%',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #f0f4f8 100%)',
        py: { xs: 4, md: 6 },
        minHeight: '100vh',
      }}
    >
      <Container maxWidth="lg">
        {/* Header Section */}
        <Grow in={true} timeout={300}>
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <Typography
              component="h1"
              variant="h2"
              sx={{
                mb: 2,
                color: '#1F2937',
                fontWeight: 700,
              }}
            >
              About CVD Risk
            </Typography>
            <Box
              sx={{
                width: 80,
                height: 4,
                background: 'linear-gradient(90deg, #5939E0 0%, #7A5FEE 100%)',
                mx: 'auto',
                borderRadius: 2,
                boxShadow: '0 4px 16px rgba(89, 57, 224, 0.3)',
                mb: 3,
              }}
            />
            <Typography
              variant="body1"
              sx={{
                color: '#6B7280',
                maxWidth: 600,
                mx: 'auto',
              }}
            >
              An advanced AI-powered platform for cardiovascular disease risk assessment using deep learning and retinal imaging analysis
            </Typography>
          </Box>
        </Grow>

        {/* About the Project Section */}
        <Grow in={true} timeout={400}>
          <Card
            sx={{
              mb: 6,
              borderRadius: 3,
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                background: 'linear-gradient(135deg, #5939E0 0%, #7A5FEE 100%)',
                p: 3,
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                gap: 2,
              }}
            >
              <LanguageIcon sx={{ fontSize: '2rem' }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                About the Platform
              </Typography>
            </Box>
            <CardContent sx={{ p: 4 }}>
              <Typography variant="body1" sx={{ mb: 2, color: '#4B5563', lineHeight: 1.8 }}>
                CVD Risk is an innovative web-based application designed to assess cardiovascular disease risk through advanced AI and machine learning models. The platform analyzes fundus (retinal) images and clinical data to provide comprehensive health insights.
              </Typography>
              <Typography variant="body1" sx={{ mb: 2, color: '#4B5563', lineHeight: 1.8 }}>
                Our mission is to democratize access to cardiovascular risk assessment tools by providing accurate, interpretable, and accessible AI-driven analysis. By leveraging deep learning techniques, we enable early detection of cardiovascular risk factors and support clinical decision-making.
              </Typography>
              <Typography variant="body1" sx={{ color: '#4B5563', lineHeight: 1.8 }}>
                The platform integrates four specialized models that analyze different aspects of cardiovascular health: hypertension detection, carotid artery thickness prediction, retinal vessel analysis, and comprehensive CVD risk assessment. Each model provides unique clinical insights that, when combined, offer a holistic view of a patient's cardiovascular status.
              </Typography>
            </CardContent>
          </Card>
        </Grow>

        {/* Team Section */}
        <Grow in={true} timeout={500}>
          <Box sx={{ mb: 6 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 4 }}>
              <GroupIcon sx={{ fontSize: '2rem', color: '#5939E0' }} />
              <Typography variant="h3" sx={{ color: '#1F2937', fontWeight: 700 }}>
                Our Team
              </Typography>
            </Box>
            <Grid container spacing={3}>
              {teamMembers.map((member, index) => (
                <Grow in={true} timeout={400 + index * 100} key={index}>
                  <Grid item xs={12} sm={6} md={3}>
                    <Card
                      sx={{
                        borderRadius: 3,
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
                        height: '100%',
                        transition: 'all 0.3s ease',
                        border: '2px solid #5939E0',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: '0 8px 24px rgba(89, 57, 224, 0.15)',
                        },
                      }}
                    >
                      <CardContent
                        sx={{
                          textAlign: 'center',
                          p: 4,
                          background: 'linear-gradient(135deg, rgba(89, 57, 224, 0.05) 0%, rgba(122, 95, 238, 0.05) 100%)',
                        }}
                      >
                        <Box
                          sx={{
                            width: 64,
                            height: 64,
                            background: 'linear-gradient(135deg, #5939E0 0%, #7A5FEE 100%)',
                            borderRadius: '50%',
                            mx: 'auto',
                            mb: 2,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'white',
                            fontSize: '1.5rem',
                            fontWeight: 700,
                          }}
                        >
                          {member.name.charAt(0)}
                        </Box>
                        <Typography variant="h6" sx={{ mb: 1, color: '#1F2937', fontWeight: 700 }}>
                          {member.name}
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#5939E0', fontWeight: 600 }}>
                          {member.role}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grow>
              ))}
            </Grid>
          </Box>
        </Grow>

        <Divider sx={{ my: 6, borderColor: 'rgba(89, 57, 224, 0.2)' }} />

        {/* Models Section */}
        <Grow in={true} timeout={600}>
          <Box>
            <Typography variant="h3" sx={{ color: '#1F2937', fontWeight: 700, mb: 4 }}>
              Our Models
            </Typography>
            <Grid container spacing={3}>
              {models.map((model, index) => (
                <Grow in={true} timeout={500 + index * 100} key={index}>
                  <Grid item xs={12} md={6}>
                    <Card
                      sx={{
                        borderRadius: 3,
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
                        height: '100%',
                        overflow: 'hidden',
                        transition: 'all 0.3s ease',
                        border: `2px solid ${model.color}`,
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: `0 8px 24px ${model.color}20`,
                        },
                      }}
                    >
                      <Box
                        sx={{
                          background: model.bgColor,
                          p: 3,
                          display: 'flex',
                          alignItems: 'center',
                          gap: 2,
                        }}
                      >
                        <Box
                          sx={{
                            width: 48,
                            height: 48,
                            background: model.color,
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'white',
                          }}
                        >
                          <model.icon sx={{ fontSize: '1.5rem' }} />
                        </Box>
                        <Box>
                          <Typography variant="h5" sx={{ color: '#1F2937', fontWeight: 700 }}>
                            {model.title}
                          </Typography>
                          <Typography variant="body2" sx={{ color: '#6B7280' }}>
                            {model.description}
                          </Typography>
                        </Box>
                      </Box>
                      <CardContent sx={{ p: 3 }}>
                        <ul style={{ margin: 0, paddingLeft: 20, color: '#4B5563' }}>
                          {model.details.map((detail, i) => (
                            <li key={i} style={{ marginBottom: 8, lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                              <Typography variant="body2" component="span">
                                {detail}
                              </Typography>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grow>
              ))}
            </Grid>
          </Box>
        </Grow>

        {/* Footer Note */}
        <Box sx={{ mt: 8, textAlign: 'center' }}>
          <Typography variant="body2" sx={{ color: '#9CA3AF' }}>
            This platform is intended for research and educational purposes. Always consult with healthcare professionals for medical decisions.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default AboutPage;
