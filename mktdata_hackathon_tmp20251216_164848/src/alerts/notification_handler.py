"""
Notification handler for alert delivery.
Supports webhook, email, and SMS notifications.
"""

from typing import List, Dict, Optional
from datetime import datetime
import requests
import structlog

from src.alerts.alert_engine import Alert, AlertSeverity
from src.utils.config import settings

logger = structlog.get_logger(__name__)


class NotificationHandler:
    """
    Handles delivery of alerts through multiple channels.
    
    Supports webhooks, email (via SMTP), and SMS (via Twilio).
    """
    
    def __init__(self):
        """Initialize notification handler."""
        self.logger = logger
        self.webhook_urls: List[str] = []
        self.enabled_channels = {
            "webhook": True,
            "email": False,  # Requires SMTP configuration
            "sms": bool(settings.twilio_account_sid),  # Enable if Twilio configured
        }
    
    def send_webhook(self, alert: Alert, webhook_url: str) -> bool:
        """
        Send alert via webhook.
        
        Args:
            alert: Alert object
            webhook_url: Webhook URL
            
        Returns:
            True if successful, False otherwise
        """
        try:
            payload = alert.to_dict()
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            
            logger.info("Webhook notification sent", alert_id=alert.alert_id, url=webhook_url)
            return True
        
        except Exception as e:
            logger.error("Webhook notification failed", error=str(e), url=webhook_url)
            return False
    
    def send_email(self, alert: Alert, recipient: str) -> bool:
        """
        Send alert via email (placeholder - requires SMTP configuration).
        
        Args:
            alert: Alert object
            recipient: Email address
            
        Returns:
            True if successful, False otherwise
        """
        # Placeholder - implement with SMTP library (smtplib)
        logger.warning("Email notification not implemented", alert_id=alert.alert_id)
        return False
    
    def send_sms(self, alert: Alert, phone_number: str) -> bool:
        """
        Send alert via SMS using Twilio.
        
        Args:
            alert: Alert object
            phone_number: Phone number (E.164 format)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from twilio.rest import Client
            
            if not settings.twilio_account_sid or not settings.twilio_auth_token:
                logger.warning("Twilio not configured")
                return False
            
            client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
            
            message = f"[{alert.severity.value.upper()}] {alert.symbol}: {alert.message}"
            
            client.messages.create(
                body=message,
                from_=settings.twilio_phone_number,
                to=phone_number,
            )
            
            logger.info("SMS notification sent", alert_id=alert.alert_id, phone=phone_number)
            return True
        
        except Exception as e:
            logger.error("SMS notification failed", error=str(e))
            return False
    
    def notify(self, alert: Alert, channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Send alert through specified channels.
        
        Args:
            alert: Alert object
            channels: List of channels ('webhook', 'email', 'sms')
                    If None, uses enabled channels based on severity
            
        Returns:
            Dictionary with channel success status
        """
        if channels is None:
            # Auto-select channels based on severity
            if alert.severity == AlertSeverity.CRITICAL:
                channels = ["webhook", "sms"]
            elif alert.severity == AlertSeverity.HIGH:
                channels = ["webhook"]
            else:
                channels = ["webhook"]
        
        results = {}
        
        for channel in channels:
            if not self.enabled_channels.get(channel, False):
                logger.debug(f"Channel {channel} not enabled, skipping")
                results[channel] = False
                continue
            
            if channel == "webhook":
                # Send to all configured webhooks
                success = False
                for webhook_url in self.webhook_urls:
                    if self.send_webhook(alert, webhook_url):
                        success = True
                results[channel] = success
            
            elif channel == "email":
                # Would need recipient list from config
                results[channel] = False  # Not implemented
            
            elif channel == "sms":
                # Would need phone number list from config
                results[channel] = False  # Not implemented without phone numbers
        
        return results
    
    def add_webhook(self, url: str) -> None:
        """
        Add webhook URL for notifications.
        
        Args:
            url: Webhook URL
        """
        if url not in self.webhook_urls:
            self.webhook_urls.append(url)
            logger.info("Webhook added", url=url)

