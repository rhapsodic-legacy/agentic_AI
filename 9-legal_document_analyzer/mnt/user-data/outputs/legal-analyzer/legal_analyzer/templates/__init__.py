"""
Legal Document Analyzer - Templates and Standards

Standard contract templates and clauses for comparison.
"""

from typing import Optional
from ..models import DocumentType, ComplianceFramework


# =============================================================================
# Standard Terms by Document Type
# =============================================================================

STANDARD_TERMS = {
    DocumentType.NDA: {
        "confidentiality_period": "2-3 years",
        "permitted_disclosures": "employees, advisors with need to know",
        "return_of_materials": "within 30 days of termination",
        "injunctive_relief": "standard",
        "governing_law": "state of disclosing party",
        "term": "1-2 years or until purpose fulfilled",
    },
    
    DocumentType.SAAS: {
        "sla_uptime": "99.9%",
        "sla_credits": "10% for each 0.1% below SLA",
        "payment_terms": "Net 30",
        "liability_cap": "12 months of fees",
        "data_retention": "30 days post-termination",
        "security_standards": "SOC 2 Type II",
        "termination_notice": "30 days",
        "auto_renewal_notice": "30 days before renewal",
        "data_processing": "DPA required",
        "support_response": "24 hours for critical",
    },
    
    DocumentType.EMPLOYMENT: {
        "notice_period": "2 weeks",
        "non_compete": "1 year, reasonable geographic scope",
        "non_solicit": "1-2 years",
        "ip_assignment": "work product assigned to employer",
        "benefits_start": "first day or after 30 days",
        "termination": "at-will with exceptions",
        "severance": "2 weeks per year of service",
        "confidentiality": "perpetual for trade secrets",
    },
    
    DocumentType.LEASE: {
        "security_deposit": "1-2 months rent",
        "rent_increase_notice": "30-60 days",
        "maintenance_responsibility": "landlord structural, tenant minor",
        "sublease_rights": "with landlord consent",
        "termination_notice": "30-60 days",
        "late_payment_grace": "5-10 days",
        "late_fee": "5% of monthly rent",
    },
    
    DocumentType.CONSULTING: {
        "payment_terms": "Net 30",
        "ip_ownership": "client owns deliverables",
        "confidentiality": "2-3 years",
        "termination_notice": "14-30 days",
        "liability_cap": "fees paid",
        "insurance_required": "$1M professional liability",
        "non_compete": "none or very limited",
        "independent_contractor": "explicit acknowledgment",
    },
}


# =============================================================================
# Required Clauses by Document Type
# =============================================================================

REQUIRED_CLAUSES = {
    DocumentType.NDA: [
        ("Definition of Confidential Information", "critical"),
        ("Obligations of Receiving Party", "critical"),
        ("Permitted Disclosures", "critical"),
        ("Term and Termination", "critical"),
        ("Return of Materials", "recommended"),
        ("No License Granted", "recommended"),
        ("Injunctive Relief", "recommended"),
        ("Governing Law", "critical"),
    ],
    
    DocumentType.SAAS: [
        ("Service Description", "critical"),
        ("Service Level Agreement", "critical"),
        ("Payment Terms", "critical"),
        ("Data Security", "critical"),
        ("Data Processing Agreement", "critical"),
        ("Intellectual Property", "critical"),
        ("Limitation of Liability", "critical"),
        ("Indemnification", "recommended"),
        ("Termination", "critical"),
        ("Data Portability", "recommended"),
        ("Audit Rights", "recommended"),
        ("Insurance", "recommended"),
    ],
    
    DocumentType.EMPLOYMENT: [
        ("Position and Duties", "critical"),
        ("Compensation", "critical"),
        ("Benefits", "critical"),
        ("Confidentiality", "critical"),
        ("IP Assignment", "critical"),
        ("Non-Compete", "recommended"),
        ("Non-Solicitation", "recommended"),
        ("Termination", "critical"),
        ("At-Will Employment", "critical"),
        ("Dispute Resolution", "recommended"),
    ],
}


# =============================================================================
# Compliance Requirements
# =============================================================================

COMPLIANCE_REQUIREMENTS = {
    ComplianceFramework.GDPR: [
        {
            "requirement": "Data Processing Agreement",
            "description": "Controller-processor agreement with required terms",
            "keywords": ["data processing", "dpa", "controller", "processor"],
        },
        {
            "requirement": "Data Subject Rights",
            "description": "Rights to access, rectify, erase, port data",
            "keywords": ["data subject rights", "right to access", "right to erasure", "data portability"],
        },
        {
            "requirement": "International Transfers",
            "description": "Safeguards for data transfers outside EEA",
            "keywords": ["international transfer", "standard contractual clauses", "adequacy decision"],
        },
        {
            "requirement": "Security Measures",
            "description": "Technical and organizational security measures",
            "keywords": ["security measures", "encryption", "access control"],
        },
        {
            "requirement": "Breach Notification",
            "description": "72-hour notification requirement",
            "keywords": ["breach notification", "data breach", "security incident"],
        },
    ],
    
    ComplianceFramework.CCPA: [
        {
            "requirement": "Consumer Rights Notice",
            "description": "Right to know, delete, opt-out",
            "keywords": ["consumer rights", "right to know", "right to delete", "opt-out"],
        },
        {
            "requirement": "Sale of Personal Information",
            "description": "Disclosure if selling personal information",
            "keywords": ["sale of personal information", "do not sell", "selling data"],
        },
        {
            "requirement": "Service Provider Terms",
            "description": "Restrictions on use of personal information",
            "keywords": ["service provider", "business purpose", "personal information"],
        },
    ],
    
    ComplianceFramework.SOX: [
        {
            "requirement": "Internal Controls",
            "description": "Controls over financial reporting",
            "keywords": ["internal controls", "financial reporting", "audit"],
        },
        {
            "requirement": "Record Retention",
            "description": "7-year retention for audit records",
            "keywords": ["record retention", "audit records", "document retention"],
        },
        {
            "requirement": "Access Controls",
            "description": "Controls on access to financial systems",
            "keywords": ["access control", "authorization", "segregation of duties"],
        },
    ],
    
    ComplianceFramework.HIPAA: [
        {
            "requirement": "Business Associate Agreement",
            "description": "BAA with required terms for PHI",
            "keywords": ["business associate", "baa", "phi", "protected health information"],
        },
        {
            "requirement": "Security Safeguards",
            "description": "Administrative, physical, technical safeguards",
            "keywords": ["security safeguards", "hipaa security", "administrative safeguards"],
        },
        {
            "requirement": "Breach Notification",
            "description": "60-day notification requirement",
            "keywords": ["breach notification", "hipaa breach", "security incident"],
        },
    ],
}


# =============================================================================
# Risk Patterns
# =============================================================================

RISK_PATTERNS = {
    "unlimited_liability": {
        "patterns": ["unlimited liability", "no cap on liability", "liability shall not be limited"],
        "level": "critical",
        "description": "No liability cap exposes party to unlimited damages",
        "recommendation": "Negotiate a liability cap, typically 12-24 months of fees",
    },
    
    "broad_indemnification": {
        "patterns": ["indemnify and hold harmless", "defend, indemnify", "all claims arising"],
        "level": "high",
        "description": "Broad indemnification clause may expose to third-party claims",
        "recommendation": "Limit indemnification to direct breach of contract or negligence",
    },
    
    "auto_renewal": {
        "patterns": ["automatically renew", "auto-renewal", "shall renew unless"],
        "level": "medium",
        "description": "Auto-renewal may lock into unwanted contract extensions",
        "recommendation": "Negotiate shorter notice period or remove auto-renewal",
    },
    
    "unilateral_amendment": {
        "patterns": ["may modify", "reserves the right to change", "unilateral amendment"],
        "level": "high",
        "description": "One party can change terms without consent",
        "recommendation": "Require mutual consent for material changes",
    },
    
    "data_breach_carveout": {
        "patterns": ["except for data breach", "excluding security incidents", "carve out for data"],
        "level": "critical",
        "description": "Liability cap excludes data breaches, creating unlimited exposure",
        "recommendation": "Negotiate a separate, higher cap for data breach liability",
    },
    
    "short_warranty": {
        "patterns": ["warranty period of 30 days", "90 day warranty", "limited warranty"],
        "level": "medium",
        "description": "Short warranty period limits protection",
        "recommendation": "Negotiate minimum 12-month warranty",
    },
    
    "one_sided_termination": {
        "patterns": ["may terminate at any time", "sole discretion to terminate"],
        "level": "high",
        "description": "Only one party has termination rights",
        "recommendation": "Ensure mutual termination rights",
    },
    
    "no_audit_rights": {
        "patterns": ["no audit", "audit not permitted"],
        "level": "medium",
        "description": "Cannot verify compliance or security",
        "recommendation": "Include annual audit rights or SOC 2 report requirement",
    },
    
    "perpetual_license": {
        "patterns": ["perpetual license", "irrevocable license", "permanent rights"],
        "level": "medium",
        "description": "Grants permanent rights that survive termination",
        "recommendation": "Limit license to contract term or specific purposes",
    },
    
    "non_compete_broad": {
        "patterns": ["shall not compete", "non-compete", "competitive business"],
        "level": "high",
        "description": "Broad non-compete may be unenforceable or overly restrictive",
        "recommendation": "Limit scope, geography, and duration",
    },
}


# =============================================================================
# Sample Documents
# =============================================================================

SAMPLE_SAAS_AGREEMENT = """
SOFTWARE AS A SERVICE AGREEMENT

This Software as a Service Agreement ("Agreement") is entered into as of January 1, 2024 ("Effective Date") by and between:

CloudTech Solutions Inc., a Delaware corporation ("Provider")
and
Acme Corporation, a California corporation ("Customer")

1. DEFINITIONS
"Service" means the cloud-based software application provided by Provider.
"Customer Data" means data submitted by Customer to the Service.
"SLA" means Service Level Agreement as set forth in Exhibit A.

2. SERVICES
2.1 Provider shall provide Customer access to the Service during the Term.
2.2 Provider guarantees 99.5% uptime availability.
2.3 Support shall be provided via email with 48-hour response time.

3. FEES AND PAYMENT
3.1 Customer shall pay $50,000 annually, due Net 45.
3.2 Fees increase by 5% annually upon renewal.
3.3 Late payments accrue interest at 1.5% per month.

4. TERM AND TERMINATION
4.1 Initial term is 24 months from the Effective Date.
4.2 Agreement automatically renews for successive 12-month periods unless either party provides 90 days written notice.
4.3 Provider may terminate immediately for material breach.
4.4 Upon termination, Customer Data shall be retained for 30 days.

5. DATA SECURITY
5.1 Provider maintains SOC 2 Type I certification.
5.2 Data is encrypted at rest using AES-256.
5.3 Provider shall notify Customer of security incidents within 72 hours.

6. LIMITATION OF LIABILITY
6.1 Provider's total liability shall not exceed fees paid in the prior 6 months.
6.2 The limitation in 6.1 shall not apply to data breaches or security incidents.
6.3 Neither party shall be liable for indirect, consequential, or punitive damages.

7. INDEMNIFICATION
7.1 Customer shall indemnify Provider against all claims arising from Customer's use of the Service.
7.2 Provider shall indemnify Customer against IP infringement claims.

8. CONFIDENTIALITY
8.1 Each party shall maintain confidentiality of the other's confidential information.
8.2 Confidentiality obligations survive for 2 years after termination.

9. INTELLECTUAL PROPERTY
9.1 Provider retains all rights to the Service.
9.2 Customer retains all rights to Customer Data.
9.3 Customer grants Provider a license to use Customer Data to provide the Service.

10. GENERAL
10.1 This Agreement shall be governed by Delaware law.
10.2 Disputes shall be resolved by binding arbitration in Wilmington, Delaware.
10.3 Provider may modify this Agreement with 30 days notice.

EXHIBIT A - SERVICE LEVEL AGREEMENT
Uptime: 99.5% monthly availability
Credits: 10% credit for availability below 99.5%
Exclusions: Scheduled maintenance, force majeure
"""


SAMPLE_NDA = """
NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is effective as of March 15, 2024 ("Effective Date") between:

Innovate Labs Inc. ("Disclosing Party")
and
TechPartner Corp. ("Receiving Party")

1. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means all non-public information disclosed by the Disclosing Party, including but not limited to technical data, trade secrets, business plans, customer lists, financial information, and any information marked as "Confidential."

2. OBLIGATIONS OF RECEIVING PARTY
2.1 Receiving Party shall hold Confidential Information in strict confidence.
2.2 Receiving Party shall not disclose Confidential Information to any third party without prior written consent.
2.3 Receiving Party shall use Confidential Information solely for the Purpose of evaluating a potential business relationship.
2.4 Receiving Party shall limit access to employees with a need to know.

3. EXCLUSIONS
This Agreement does not apply to information that:
(a) Is or becomes publicly available through no fault of Receiving Party;
(b) Was known to Receiving Party prior to disclosure;
(c) Is independently developed by Receiving Party;
(d) Is disclosed pursuant to court order.

4. TERM
4.1 This Agreement shall remain in effect for 3 years from the Effective Date.
4.2 Confidentiality obligations shall survive for 5 years after termination.

5. RETURN OF MATERIALS
Upon termination or request, Receiving Party shall return or destroy all Confidential Information within 30 days.

6. NO LICENSE
Nothing in this Agreement grants any license to patents, trademarks, or other intellectual property.

7. INJUNCTIVE RELIEF
Receiving Party acknowledges that breach may cause irreparable harm and Disclosing Party shall be entitled to injunctive relief.

8. GOVERNING LAW
This Agreement shall be governed by California law.

9. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement and supersedes all prior negotiations.
"""


SAMPLE_EMPLOYMENT = """
EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into as of February 1, 2024 between:

TechCorp Inc. ("Company")
and
John Smith ("Employee")

1. POSITION AND DUTIES
1.1 Company employs Employee as Senior Software Engineer.
1.2 Employee shall report to the VP of Engineering.
1.3 Employee shall devote full working time to Company duties.

2. COMPENSATION
2.1 Base salary of $180,000 per year, paid bi-weekly.
2.2 Annual bonus of up to 20% of base salary based on performance.
2.3 Stock options of 10,000 shares vesting over 4 years with 1-year cliff.

3. BENEFITS
3.1 Health, dental, and vision insurance effective first day of employment.
3.2 401(k) with 4% company match after 90 days.
3.3 20 days paid time off annually.

4. AT-WILL EMPLOYMENT
4.1 Employment is at-will and may be terminated by either party at any time.
4.2 Company may terminate with or without cause upon 2 weeks notice.
4.3 Employee may resign with 2 weeks notice.

5. CONFIDENTIALITY
5.1 Employee shall not disclose Company's confidential information during or after employment.
5.2 Confidential information includes trade secrets, customer data, and business strategies.
5.3 This obligation survives indefinitely for trade secrets.

6. INTELLECTUAL PROPERTY
6.1 All work product created during employment belongs to Company.
6.2 Employee assigns all rights, title, and interest in inventions to Company.
6.3 Employee shall assist Company in securing IP rights.

7. NON-COMPETE
7.1 For 2 years after termination, Employee shall not work for any competitor.
7.2 Competitors include any company in the software development industry.
7.3 This restriction applies worldwide.

8. NON-SOLICITATION
8.1 For 2 years after termination, Employee shall not solicit Company employees.
8.2 Employee shall not solicit Company customers or clients.

9. TERMINATION
9.1 Upon termination, Employee shall return all Company property.
9.2 Company shall pay earned but unpaid compensation within 30 days.
9.3 No severance is provided unless required by law.

10. GOVERNING LAW
This Agreement shall be governed by California law.
"""


def get_sample_document(doc_type: DocumentType) -> str:
    """Get a sample document of the specified type."""
    samples = {
        DocumentType.SAAS: SAMPLE_SAAS_AGREEMENT,
        DocumentType.NDA: SAMPLE_NDA,
        DocumentType.EMPLOYMENT: SAMPLE_EMPLOYMENT,
    }
    return samples.get(doc_type, SAMPLE_SAAS_AGREEMENT)


def get_standard_terms(doc_type: DocumentType) -> dict:
    """Get standard terms for a document type."""
    return STANDARD_TERMS.get(doc_type, {})


def get_required_clauses(doc_type: DocumentType) -> list:
    """Get required clauses for a document type."""
    return REQUIRED_CLAUSES.get(doc_type, [])


def get_compliance_requirements(framework: ComplianceFramework) -> list:
    """Get compliance requirements for a framework."""
    return COMPLIANCE_REQUIREMENTS.get(framework, [])
