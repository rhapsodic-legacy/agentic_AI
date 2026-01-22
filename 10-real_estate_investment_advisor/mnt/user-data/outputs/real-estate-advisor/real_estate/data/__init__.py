"""
Real Estate Investment Advisor - Mock Data Sources

Simulates data from:
- Zillow, Redfin, Realtor.com (listings)
- Census Bureau, BLS (demographics)
- Rentometer, Apartments.com (rentals)
- Building department APIs (permits)
"""

from typing import Optional
from datetime import datetime, timedelta
import random
import uuid

from ..models import (
    Property, PropertyType, PropertyCondition, PropertyFeatures, Address,
    ComparableSale, MarketData, MarketMetrics, Demographics,
    RentalEstimate, LegalCheck
)


# =============================================================================
# Market Data by City
# =============================================================================

MARKET_DATA = {
    "Austin": {
        "state": "TX",
        "median_price": 550000,
        "price_change_yoy": 8.5,
        "median_rent": 2200,
        "rent_change_yoy": 6.2,
        "vacancy_rate": 5.2,
        "days_on_market": 28,
        "population": 1028225,
        "population_growth": 4.2,
        "median_income": 78000,
        "unemployment": 3.2,
        "investment_score": 82,
        "market_type": "Seller's Market",
        "price_trend": "Rising",
    },
    "Phoenix": {
        "state": "AZ",
        "median_price": 445000,
        "price_change_yoy": 5.2,
        "median_rent": 1850,
        "rent_change_yoy": 4.8,
        "vacancy_rate": 6.1,
        "days_on_market": 35,
        "population": 1680992,
        "population_growth": 3.8,
        "median_income": 65000,
        "unemployment": 3.8,
        "investment_score": 78,
        "market_type": "Balanced Market",
        "price_trend": "Rising",
    },
    "Tampa": {
        "state": "FL",
        "median_price": 395000,
        "price_change_yoy": 6.8,
        "median_rent": 1950,
        "rent_change_yoy": 7.5,
        "vacancy_rate": 4.8,
        "days_on_market": 25,
        "population": 399700,
        "population_growth": 3.5,
        "median_income": 58000,
        "unemployment": 3.5,
        "investment_score": 85,
        "market_type": "Seller's Market",
        "price_trend": "Rising",
    },
    "Denver": {
        "state": "CO",
        "median_price": 620000,
        "price_change_yoy": 3.2,
        "median_rent": 2100,
        "rent_change_yoy": 4.5,
        "vacancy_rate": 5.5,
        "days_on_market": 32,
        "population": 727211,
        "population_growth": 2.8,
        "median_income": 82000,
        "unemployment": 3.0,
        "investment_score": 72,
        "market_type": "Balanced Market",
        "price_trend": "Stable",
    },
    "Nashville": {
        "state": "TN",
        "median_price": 480000,
        "price_change_yoy": 7.2,
        "median_rent": 1900,
        "rent_change_yoy": 5.8,
        "vacancy_rate": 5.0,
        "days_on_market": 22,
        "population": 692587,
        "population_growth": 3.2,
        "median_income": 68000,
        "unemployment": 3.1,
        "investment_score": 80,
        "market_type": "Seller's Market",
        "price_trend": "Rising",
    },
    "Dallas": {
        "state": "TX",
        "median_price": 420000,
        "price_change_yoy": 4.5,
        "median_rent": 1750,
        "rent_change_yoy": 5.2,
        "vacancy_rate": 6.5,
        "days_on_market": 38,
        "population": 1304379,
        "population_growth": 2.5,
        "median_income": 60000,
        "unemployment": 3.6,
        "investment_score": 75,
        "market_type": "Balanced Market",
        "price_trend": "Stable",
    },
    "Atlanta": {
        "state": "GA",
        "median_price": 385000,
        "price_change_yoy": 5.8,
        "median_rent": 1800,
        "rent_change_yoy": 6.0,
        "vacancy_rate": 5.8,
        "days_on_market": 30,
        "population": 498715,
        "population_growth": 2.2,
        "median_income": 62000,
        "unemployment": 3.9,
        "investment_score": 77,
        "market_type": "Balanced Market",
        "price_trend": "Rising",
    },
}

# Sample street names
STREET_NAMES = [
    "Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Pine St",
    "Elm St", "Park Ave", "Lake Dr", "Hill Rd", "Valley Way",
    "Sunset Blvd", "Mountain View", "River Rd", "Forest Dr", "Meadow Ln"
]


# =============================================================================
# Property Data Source
# =============================================================================

class PropertyDataSource:
    """
    Mock property data source (simulates Zillow/Redfin APIs).
    """
    
    def __init__(self):
        self._properties = {}
        self._generate_sample_properties()
    
    def _generate_sample_properties(self):
        """Generate sample properties for each market."""
        for city, data in MARKET_DATA.items():
            for i in range(5):
                prop = self._create_property(city, data, i)
                self._properties[prop.property_id] = prop
    
    def _create_property(self, city: str, market: dict, index: int) -> Property:
        """Create a sample property."""
        base_price = market["median_price"]
        
        # Vary properties
        price_multiplier = random.uniform(0.7, 1.4)
        price = int(base_price * price_multiplier)
        
        bedrooms = random.choice([2, 3, 3, 3, 4, 4, 5])
        bathrooms = bedrooms - random.choice([0, 0.5, 1])
        sqft = bedrooms * random.randint(400, 600) + random.randint(200, 500)
        
        prop_type = random.choice([
            PropertyType.SINGLE_FAMILY,
            PropertyType.SINGLE_FAMILY,
            PropertyType.SINGLE_FAMILY,
            PropertyType.TOWNHOUSE,
            PropertyType.CONDO,
        ])
        
        return Property(
            property_id=f"prop-{uuid.uuid4().hex[:8]}",
            address=Address(
                street=f"{random.randint(100, 9999)} {random.choice(STREET_NAMES)}",
                city=city,
                state=market["state"],
                zip_code=f"{random.randint(70000, 89999)}",
            ),
            property_type=prop_type,
            features=PropertyFeatures(
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                sqft=sqft,
                lot_sqft=sqft * random.randint(2, 5) if prop_type == PropertyType.SINGLE_FAMILY else 0,
                year_built=random.randint(1980, 2023),
                stories=random.choice([1, 1, 2, 2, 2]),
                garage_spaces=random.choice([0, 1, 2, 2, 2, 3]),
                pool=random.random() < 0.2,
                hoa=prop_type in [PropertyType.CONDO, PropertyType.TOWNHOUSE],
                hoa_fee=random.choice([150, 200, 250, 300, 350]) if prop_type in [PropertyType.CONDO, PropertyType.TOWNHOUSE] else 0,
            ),
            list_price=price,
            condition=random.choice(list(PropertyCondition)),
            days_on_market=random.randint(1, 90),
            listing_date=(datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
        )
    
    def search_properties(
        self,
        city: str = None,
        min_price: float = 0,
        max_price: float = float('inf'),
        min_beds: int = 0,
        property_type: PropertyType = None,
    ) -> list[Property]:
        """Search for properties."""
        results = []
        
        for prop in self._properties.values():
            if city and prop.address.city.lower() != city.lower():
                continue
            if prop.list_price < min_price or prop.list_price > max_price:
                continue
            if prop.features.bedrooms < min_beds:
                continue
            if property_type and prop.property_type != property_type:
                continue
            
            results.append(prop)
        
        return results
    
    def get_property(self, property_id: str) -> Optional[Property]:
        """Get a property by ID."""
        return self._properties.get(property_id)
    
    def get_comparable_sales(self, property: Property, radius_miles: float = 1.0) -> list[ComparableSale]:
        """Get comparable sales for a property."""
        comps = []
        base_price = property.list_price
        
        for i in range(5):
            # Generate similar properties
            price_var = random.uniform(-0.15, 0.15)
            sqft_var = random.uniform(-0.1, 0.1)
            
            sale_price = int(base_price * (1 + price_var))
            sqft = int(property.features.sqft * (1 + sqft_var))
            
            days_ago = random.randint(30, 180)
            
            comps.append(ComparableSale(
                address=f"{random.randint(100, 9999)} {random.choice(STREET_NAMES)}, {property.address.city}",
                sale_price=sale_price,
                sale_date=(datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
                sqft=sqft,
                bedrooms=property.features.bedrooms + random.choice([-1, 0, 0, 0, 1]),
                bathrooms=property.features.bathrooms + random.choice([-0.5, 0, 0, 0, 0.5]),
                distance_miles=random.uniform(0.2, radius_miles),
                similarity_score=random.randint(75, 95),
            ))
        
        return sorted(comps, key=lambda c: c.similarity_score, reverse=True)
    
    def add_property(self, property: Property):
        """Add a property to the database."""
        self._properties[property.property_id] = property


# =============================================================================
# Market Data Source
# =============================================================================

class MarketDataSource:
    """
    Mock market data source (simulates Census/BLS/MLS APIs).
    """
    
    def get_market_data(self, city: str) -> Optional[MarketData]:
        """Get market data for a city."""
        city_key = city.title()
        
        if city_key not in MARKET_DATA:
            # Return default data for unknown cities
            return self._create_default_market_data(city)
        
        data = MARKET_DATA[city_key]
        
        return MarketData(
            city=city_key,
            state=data["state"],
            zip_code="",
            metrics=MarketMetrics(
                median_price=data["median_price"],
                median_price_sqft=data["median_price"] / 1800,
                price_change_yoy=data["price_change_yoy"],
                active_listings=random.randint(500, 2000),
                months_supply=random.uniform(1.5, 4.0),
                days_on_market_avg=data["days_on_market"],
                sales_last_month=random.randint(200, 800),
                sales_yoy_change=random.uniform(-5, 15),
                median_rent=data["median_rent"],
                rent_change_yoy=data["rent_change_yoy"],
                vacancy_rate=data["vacancy_rate"],
            ),
            demographics=Demographics(
                population=data["population"],
                population_growth=data["population_growth"],
                median_household_income=data["median_income"],
                income_growth=random.uniform(2, 5),
                unemployment_rate=data["unemployment"],
                median_age=random.uniform(32, 40),
                college_educated_pct=random.uniform(30, 50),
                owner_occupied_pct=random.uniform(45, 65),
                renter_occupied_pct=random.uniform(35, 55),
            ),
            price_trend=data["price_trend"],
            rent_trend="Rising" if data["rent_change_yoy"] > 3 else "Stable",
            market_type=data["market_type"],
            investment_score=data["investment_score"],
            growth_score=int(data["population_growth"] * 20),
            affordability_score=max(20, 100 - int(data["median_price"] / 10000)),
        )
    
    def _create_default_market_data(self, city: str) -> MarketData:
        """Create default market data for unknown cities."""
        return MarketData(
            city=city,
            state="XX",
            zip_code="",
            metrics=MarketMetrics(
                median_price=350000,
                median_price_sqft=200,
                price_change_yoy=3.0,
                active_listings=500,
                months_supply=3.0,
                days_on_market_avg=35,
                sales_last_month=150,
                sales_yoy_change=0,
                median_rent=1500,
                rent_change_yoy=3.0,
                vacancy_rate=6.0,
            ),
            demographics=Demographics(
                population=100000,
                population_growth=1.5,
                median_household_income=55000,
                income_growth=2.5,
                unemployment_rate=4.5,
                median_age=36,
                college_educated_pct=35,
                owner_occupied_pct=55,
                renter_occupied_pct=45,
            ),
            investment_score=60,
        )


# =============================================================================
# Rental Data Source
# =============================================================================

class RentalDataSource:
    """
    Mock rental data source (simulates Rentometer/Apartments.com APIs).
    """
    
    def estimate_rent(self, property: Property) -> RentalEstimate:
        """Estimate rental income for a property."""
        city = property.address.city.title()
        
        # Get market data
        market = MARKET_DATA.get(city, {"median_rent": 1500, "rent_change_yoy": 3.0})
        base_rent = market["median_rent"]
        
        # Adjust for property characteristics
        bed_adjustment = (property.features.bedrooms - 3) * 200
        sqft_adjustment = (property.features.sqft - 1500) * 0.3
        
        # Condition adjustment
        condition_mult = {
            PropertyCondition.EXCELLENT: 1.15,
            PropertyCondition.GOOD: 1.05,
            PropertyCondition.FAIR: 0.95,
            PropertyCondition.POOR: 0.85,
            PropertyCondition.NEEDS_RENOVATION: 0.75,
        }
        
        multiplier = condition_mult.get(property.condition, 1.0)
        
        # Calculate estimate
        estimated_rent = (base_rent + bed_adjustment + sqft_adjustment) * multiplier
        estimated_rent = max(800, int(estimated_rent / 50) * 50)  # Round to $50
        
        variance = 0.1
        
        return RentalEstimate(
            monthly_rent=estimated_rent,
            low_estimate=int(estimated_rent * (1 - variance)),
            high_estimate=int(estimated_rent * (1 + variance)),
            market_average=base_rent,
            percentile=min(95, max(5, 50 + int((estimated_rent - base_rent) / 50))),
            confidence=random.uniform(75, 95),
            comparable_rentals=random.randint(10, 50),
        )
    
    def get_rental_comps(self, property: Property) -> list[dict]:
        """Get comparable rentals."""
        city = property.address.city.title()
        market = MARKET_DATA.get(city, {"median_rent": 1500})
        base_rent = market["median_rent"]
        
        comps = []
        for i in range(5):
            rent_var = random.uniform(-0.15, 0.15)
            
            comps.append({
                "address": f"{random.randint(100, 9999)} {random.choice(STREET_NAMES)}",
                "rent": int((base_rent * (1 + rent_var)) / 50) * 50,
                "bedrooms": property.features.bedrooms + random.choice([-1, 0, 0, 1]),
                "sqft": property.features.sqft + random.randint(-200, 200),
                "distance": round(random.uniform(0.2, 1.5), 1),
            })
        
        return comps


# =============================================================================
# Legal Data Source
# =============================================================================

class LegalDataSource:
    """
    Mock legal/permit data source.
    """
    
    def check_zoning(self, property: Property) -> dict:
        """Check zoning compliance."""
        zones = ["R-1 Residential", "R-2 Residential", "R-3 Multi-Family", "MU Mixed Use"]
        
        return {
            "zoning_code": random.choice(["R-1", "R-2", "R-3", "MU"]),
            "zoning_description": random.choice(zones),
            "residential_allowed": True,
            "rental_allowed": True,
            "short_term_rental_allowed": random.random() < 0.3,
            "max_units": random.choice([1, 2, 4]),
        }
    
    def check_permits(self, property: Property) -> list[dict]:
        """Check building permits."""
        permits = []
        
        # Random chance of open permits
        if random.random() < 0.2:
            permits.append({
                "permit_number": f"BLD-{random.randint(10000, 99999)}",
                "type": random.choice(["Electrical", "Plumbing", "HVAC", "Addition"]),
                "status": "Open",
                "issued_date": (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
            })
        
        return permits
    
    def check_title(self, property: Property) -> dict:
        """Check title status."""
        has_lien = random.random() < 0.1
        
        return {
            "title_clear": not has_lien,
            "liens": [{"type": "Tax Lien", "amount": random.randint(5000, 20000)}] if has_lien else [],
            "easements": random.random() < 0.3,
            "hoa_violations": [],
        }
    
    def get_legal_checks(self, property: Property) -> list[LegalCheck]:
        """Get all legal checks."""
        checks = []
        
        # Zoning
        zoning = self.check_zoning(property)
        checks.append(LegalCheck(
            item="Zoning",
            status="clear",
            details=f"{zoning['zoning_code']} - {zoning['zoning_description']}. Rental allowed.",
        ))
        
        # Permits
        permits = self.check_permits(property)
        if permits:
            checks.append(LegalCheck(
                item="Building Permits",
                status="warning",
                details=f"{len(permits)} open permit(s) found. Verify completion.",
            ))
        else:
            checks.append(LegalCheck(
                item="Building Permits",
                status="clear",
                details="No open permits.",
            ))
        
        # Title
        title = self.check_title(property)
        if title["title_clear"]:
            checks.append(LegalCheck(
                item="Title",
                status="clear",
                details="Title clear. No liens found.",
            ))
        else:
            checks.append(LegalCheck(
                item="Title",
                status="issue",
                details=f"Lien found: {title['liens'][0]['type']} - ${title['liens'][0]['amount']:,}",
            ))
        
        # HOA (if applicable)
        if property.features.hoa:
            checks.append(LegalCheck(
                item="HOA",
                status="clear" if random.random() > 0.2 else "warning",
                details=f"HOA Fee: ${property.features.hoa_fee}/month. " + 
                       ("Rental restrictions may apply." if random.random() < 0.3 else "No rental restrictions."),
            ))
        
        return checks


# =============================================================================
# Global Instances
# =============================================================================

property_source = PropertyDataSource()
market_source = MarketDataSource()
rental_source = RentalDataSource()
legal_source = LegalDataSource()
