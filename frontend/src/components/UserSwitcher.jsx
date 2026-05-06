import React from 'react';
import {
  Menu, MenuTrigger, MenuPopover, MenuList, MenuItem,
  makeStyles, Spinner,
} from '@fluentui/react-components';
import { SignOut20Regular, Person16Regular, Checkmark16Regular } from '@fluentui/react-icons';
import { useUser } from '../context/UserContext';

const useStyles = makeStyles({
  trigger: {
    width: '30px',
    height: '30px',
    borderRadius: '50%',
    border: 'none',
    background: 'linear-gradient(135deg, #e0e7ff, #c7d2fe)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    fontSize: '12px',
    fontWeight: '600',
    color: '#4338ca',
    fontFamily: "'DM Sans', sans-serif",
    transition: 'opacity 0.15s',
    ':hover': {
      opacity: 0.8,
    },
  },
  itemLabel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1px',
  },
  itemName: {
    fontSize: '13px',
    fontFamily: "'DM Sans', sans-serif",
  },
  itemEmail: {
    fontSize: '11px',
    color: '#9ca3af',
    fontFamily: "'DM Sans', sans-serif",
  },
});

function initials(name) {
  if (!name) return '?';
  return name.split(' ').map(p => p[0]).join('').toUpperCase().slice(0, 2);
}

export default function UserSwitcher({ onLogout }) {
  const styles = useStyles();
  const { activeUser, activeUserId, setActiveUserId, users, loading } = useUser();

  if (loading) return <Spinner size="tiny" />;

  const displayName = activeUser?.display_name || activeUserId || 'User';

  return (
    <Menu>
      <MenuTrigger disableButtonEnhancement>
        <button className={styles.trigger} title={displayName}>
          {initials(displayName)}
        </button>
      </MenuTrigger>
      <MenuPopover>
        <MenuList>
          {(users || []).map(u => (
            <MenuItem
              key={u.id}
              onClick={() => setActiveUserId(u.id)}
              icon={u.id === activeUserId ? <Checkmark16Regular style={{ color: '#4f46e5' }} /> : <Person16Regular />}
            >
              <div className={styles.itemLabel}>
                <span className={styles.itemName} style={u.id === activeUserId ? { fontWeight: 600, color: '#4f46e5' } : {}}>
                  {u.display_name || u.id}
                </span>
                {u.email && <span className={styles.itemEmail}>{u.email}</span>}
              </div>
            </MenuItem>
          ))}
          {onLogout && (
            <MenuItem icon={<SignOut20Regular />} onClick={onLogout}>
              <span style={{ fontFamily: "'DM Sans', sans-serif", fontSize: '13px' }}>Sign out</span>
            </MenuItem>
          )}
        </MenuList>
      </MenuPopover>
    </Menu>
  );
}
