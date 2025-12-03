/**
 * Home page - Landing page with model selection
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Grow,
  Tooltip,
  IconButton,
  Switch,
} from '@mui/material';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import TimelineIcon from '@mui/icons-material/Timeline';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import HubIcon from '@mui/icons-material/Hub';

interface HomePageProps {
  apiReady: boolean;
}

const HomePage: React.FC<HomePageProps> = ({ apiReady }) => {
  const navigate = useNavigate();
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [darkMode, setDarkMode] = useState(false);

  const models = [
    {
      title: 'Hypertension Detection',
      description: 'Binary classification of hypertensive retinopathy',
      subtitle: 'RETFound ViT Model',
      icon: FavoriteBorderIcon,
      path: '/htn',
      color: '#ff6f61', // soft coral
      bgColor: '#ffeaea',
    },
    {
      title: 'CIMT Regression',
      description: 'Predict carotid intima-media thickness (0.4 - 1.2)',
      subtitle: 'Siamese Multimodal',
      icon: TimelineIcon,
      path: '/cimt',
      color: '#64b5f6', // soft blue
      bgColor: '#e3f2fd',
    },
    {
      title: 'A/V Segmentation',
      description: 'Segment retinal blood vessels and extract features',
      subtitle: 'U-Net Architecture',
      icon: AccountTreeIcon,
      path: '/vessel',
      color: '#81c784', // soft green
      bgColor: '#e8f5e9',
    },
    {
      title: 'Fusion Model',
      description: 'Complete CVD risk assessment from all models',
      subtitle: 'Meta-Classifier (MLP)',
      icon: HubIcon,
      path: '/fusion',
      color: '#b39ddb', // soft purple
      bgColor: '#f3e5f5',
    },
  ];


  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        minHeight: '100%',
        flex: 1,
        background: darkMode ? '#1a1a1a' : '#ffffff',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        overflow: 'hidden',
        transition: 'background 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      {/* Theme Toggle */}
      <Box sx={{ position: 'absolute', top: 24, right: 32, zIndex: 10, display: 'flex', alignItems: 'center', gap: 1 }}>
        <IconButton
          onClick={() => setDarkMode((d) => !d)}
          size="large"
          sx={{
            color: darkMode ? '#9D7FFF' : '#6A4DF5',
            transition: 'all 0.3s ease',
            '&:hover': {
              backgroundColor: darkMode ? 'rgba(166, 128, 255, 0.1)' : 'rgba(106, 77, 245, 0.1)',
              transform: 'scale(1.05)',
            },
          }}
        >
          {darkMode ? <LightModeIcon sx={{ fontSize: '1.5rem' }} /> : <DarkModeIcon sx={{ fontSize: '1.5rem' }} />}
        </IconButton>
        <Switch
          checked={darkMode}
          onChange={() => setDarkMode((d) => !d)}
          sx={{
            '& .MuiSwitch-switchBase.Mui-checked': {
              color: '#9D7FFF',
              '&:hover': { backgroundColor: 'rgba(166, 128, 255, 0.1)' },
            },
            '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
              backgroundColor: '#9D7FFF',
            },
          }}
        />
      </Box>
      {/* Main Content */}
      <Box sx={{ flex: 1, width: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', pt: { xs: 3, md: 5 }, pb: { xs: 3, md: 4 } }}>
        {/* Header */}
        <Box sx={{ textAlign: 'center', mb: 4, px: { xs: 3, md: 2 }, width: '100%' }}>
          <Typography
            component="h1"
            variant="h1"
            sx={{
              mb: 2,
              color: darkMode ? '#E8DAFF' : '#1F2937',
            }}
          >
            Cardiovascular Risk Assessment
          </Typography>
          <Typography
            variant="body1"
            sx={{
              color: darkMode ? '#B8A0D0' : '#6B7280',
              mb: 3,
              maxWidth: 600,
              mx: 'auto',
            }}
          >
            Advanced AI-Powered Analysis from Retinal Imaging
          </Typography>
          <Box
            sx={{
              width: 80,
              height: 4,
              background: darkMode
                ? 'linear-gradient(90deg, #9D7FFF 0%, #7A5FEE 100%)'
                : 'linear-gradient(90deg, #5939E0 0%, #7A5FEE 100%)',
              mx: 'auto',
              borderRadius: 2,
              boxShadow: darkMode
                ? '0 4px 16px rgba(157, 127, 255, 0.3)'
                : '0 4px 16px rgba(89, 57, 224, 0.3)',
            }}
          />
        </Box>
        {/* Model Selection Grid */}
        <Grid container spacing={3} sx={{ mb: 4, px: { xs: 2, sm: 3, md: 4 }, width: '100%', maxWidth: 1400, justifyContent: 'center' }}>
          {models.map((model, index) => (
            <Grow in={true} timeout={300 + index * 100} key={index}>
              <Grid item xs={12} sm={6} md={3} lg={3} xl={3} sx={{ display: 'flex', justifyContent: 'center' }}>
                <Tooltip title={apiReady ? 'Click to analyze' : 'Loading...'} arrow>
                  <Card
                    onMouseEnter={() => setHoveredIndex(index)}
                    onMouseLeave={() => setHoveredIndex(null)}
                    onClick={() => apiReady && navigate(model.path)}
                    sx={{
                      cursor: apiReady ? 'pointer' : 'not-allowed',
                      width: '100%',
                      maxWidth: { xs: 260, sm: 280, md: 320 },
                      height: { xs: 260, sm: 280, md: 320 },
                      background: darkMode ? '#2a2a2a' : '#ffffff',
                      border: `2px solid #5939E0`,
                      borderRadius: 10,
                      boxShadow: hoveredIndex === index
                        ? `0 8px 24px rgba(89, 57, 224, 0.12)`
                        : '0 1px 3px rgba(0,0,0,0.08)',
                      opacity: apiReady ? 1 : 0.5,
                      position: 'relative',
                      overflow: 'hidden',
                      transition: 'all 0.15s cubic-bezier(0.23, 1, 0.32, 1)',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'stretch',
                      justifyContent: 'center',
                      '&:hover': apiReady ? {
                        transform: 'translateY(-4px)',
                        boxShadow: `0 8px 24px rgba(89, 57, 224, 0.15)`,
                        borderColor: '#4A2DB0',
                      } : {},
                    }}
                  >
                    <CardContent sx={{ textAlign: 'center', p: { xs: 3, md: 4 }, position: 'relative', zIndex: 1, width: '100%', display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'space-between' }}>
                      <Box
                        sx={{
                          mb: 2,
                          color: '#5939E0',
                          transition: 'all 0.15s ease',
                          transform: hoveredIndex === index ? 'scale(1.1) translateY(-4px)' : 'scale(1)',
                          display: 'flex',
                          justifyContent: 'center',
                        }}
                      >
                        <model.icon sx={{ fontSize: { xs: '2.4rem', sm: '2.8rem', md: '3.2rem' } }} />
                      </Box>
                      <Box sx={{ flex: 1 }}>
                        <Typography
                          component="h2"
                          variant="h5"
                          sx={{
                            mb: 1,
                            color: darkMode ? '#E8DAFF' : '#1F2937',
                          }}
                        >
                          {model.title}
                        </Typography>
                        <Typography
                          variant="caption"
                          sx={{
                            color: darkMode ? 'rgba(230, 200, 255, 0.7)' : '#5939E0',
                            display: 'block',
                            mb: 2,
                            textTransform: 'uppercase',
                          }}
                        >
                          {model.subtitle}
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{
                            color: darkMode ? '#C5B0E0' : '#666',
                            lineHeight: 1.6,
                            minHeight: 44,
                            fontSize: { xs: '0.9rem', sm: '0.95rem', md: '1rem' },
                            fontWeight: 500,
                          }}
                      >
                        {model.description}
                      </Typography>
                    </Box>
                    </CardContent>
                  </Card>
                </Tooltip>
              </Grid>
            </Grow>
          ))}
        </Grid>
      </Box>
    </Box>
  );
};

export default HomePage;
